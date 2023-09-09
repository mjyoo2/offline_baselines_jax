from dataclasses import dataclass
import jax.numpy as jnp

@dataclass(frozen=True)
class DDPMCoefficients:
    alpha_t: jnp.ndarray
    oneover_sqrta: jnp.ndarray
    sqrt_beta_t: jnp.ndarray
    alpha_bar_t: jnp.ndarray
    sqrtab: jnp.ndarray
    sqrtmab: jnp.ndarray
    mab_over_sqrtmab_inv: jnp.ndarray
    ma_over_sqrtmab_inv: jnp.ndarray


class DiffusionBetaScheduler:
    supported_schedulers = ["linear", "cosine"]
    def __init__(
            self,
            beta1: float = 1e-4,
            beta2: float = 0.02,
            total_denoise_steps: int = 8,
            method: str = "cosine"
    ):
        self.beta1 = beta1
        self.beta2 = beta2
        self.total_denoise_steps = total_denoise_steps
        self.method = method.lower()
        assert method in DiffusionBetaScheduler.supported_schedulers, f"{method} is not supported beta scheduler."

    def schedule(self) -> DDPMCoefficients:
        if self.method == "linear":
            beta_t = self.linear_schedule()
        elif self.method == "cosine":
            beta_t = self.cosine_schedule()
        else:
            raise NotImplementedError(f"{self.method} is not supported beta scheduler.")

        sqrt_beta_t = jnp.sqrt(beta_t)
        alpha_t = 1 - beta_t
        log_alpha_t = jnp.log(alpha_t)
        alpha_bar_t = jnp.exp(jnp.cumsum(log_alpha_t, axis=0))  # = alphas_cumprod

        sqrtab = jnp.sqrt(alpha_bar_t)
        oneover_sqrta = 1 / jnp.sqrt(alpha_t)

        sqrtmab = jnp.sqrt(1 - alpha_bar_t)
        mab_over_sqrtmab_inv = (1 - alpha_bar_t) / sqrtmab
        ma_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

        return DDPMCoefficients(
            alpha_t=alpha_t,
            oneover_sqrta=oneover_sqrta,
            sqrt_beta_t=sqrt_beta_t,
            alpha_bar_t=alpha_bar_t,
            sqrtab=sqrtab,
            sqrtmab=sqrtmab,
            mab_over_sqrtmab_inv=mab_over_sqrtmab_inv,
            ma_over_sqrtmab_inv=ma_over_sqrtmab_inv
        )

    def linear_schedule(self) -> jnp.ndarray:
        beta_t = (self.beta2 - self.beta1) \
                 * jnp.arange(-1, self.total_denoise_steps, dtype=jnp.float32) \
                 / (self.total_denoise_steps - 1) \
                 + self.beta1
        beta_t = beta_t.at[0].set(self.beta1)

        return beta_t

    def cosine_schedule(self):
        s = 8e-3
        timesteps = jnp.arange(self.total_denoise_steps + 1, dtype=jnp.float32)
        x = (((timesteps / self.total_denoise_steps) + s) / (1 + s)) * (jnp.pi / 2)
        f_t = jnp.cos(x) ** 2

        x_0 = (s / (1 + s)) * (jnp.pi / 2)
        f_0 = jnp.cos(x_0) ** 2

        alpha_bar_t = f_t / f_0

        beta_t = 1 - alpha_bar_t[1:] / alpha_bar_t[: -1]
        beta_t = jnp.clip(beta_t, a_min=0, a_max=0.999)

        return beta_t