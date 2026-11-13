title = "Likelihood"
description = "Arguments affecting the likelihood term in the ELBO estimated by ABISMAL"

from abismal.command_line.parser.custom_types import degrees_of_freedom

args_and_kwargs=(
    (
        (
            "-t",
            "--studentt-dof",
        ),{
            "help": "Degrees of freedom for the t-distributed error model, default 32. A t "
                    "distribution converges to a normal as this goes to infinity, so pass 0, "
                    "inf or none for a normal error model.",
            "type" : degrees_of_freedom,
            "default": 32.,
        }
    ),
    (
        (
            "--error-model",
        ),{
            "help": "'ev11' learns the Evans (2011) error model Var(I) = a·σ² + b²·(s·Î)² with the rest of the model, "
                    "s·Î the predicted intensity (detached in the variance). The default, 'none', uses the reported "
                    "uncertainties.",
            "default": "none",
            "choices": ["none", "ev11"],
        }
    ),
    (
        (
            "--no-error-model-detach",
        ),{
            "help": "--error-model ev11: let the predicted intensity in the variance carry gradient to the scales and "
                    "the surrogate posterior (detached by default).",
            "dest": "error_model_detach",
            "action": "store_false",
        }
    ),
    (
        (
            "--error-model-a",
        ),{
            "help": "--error-model ev11: the starting (or, with --fix-error-model, fixed) a. Default 1.",
            "default": 1., "type": float,
        }
    ),
    (
        (
            "--error-model-b",
        ),{
            "help": "--error-model ev11: the starting (or, with --fix-error-model, fixed) b. Default 0.01.",
            "default": 0.01, "type": float,
        }
    ),
    (
        (
            "--fix-error-model",
        ),{
            "help": "--error-model ev11: hold a and b at --error-model-a and --error-model-b instead of learning them.",
            "action": "store_true",
        }
    ),
    (
        (
            "--error-model-fit",
        ),{
            "help": "--error-model ev11: how to learn a and b. 'elbo' (default): with the rest of the model, on the ELBO. "
                    "'online': outside the ELBO, in closed form, from running sums of a weighted least-squares fit of "
                    "the squared residuals against the mean prediction, updated every training step. "
                    "'normal': on the ELBO but with a normal likelihood, while the Student t trains the rest of the "
                    "model (needs a Student t).",
            "default": "elbo",
            "choices": ["elbo", "online", "normal"],
        }
    ),
    (
        (
            "--error-model-form",
        ),{
            "help": "--error-model ev11: the variance's form, with p = s*I_pred the predicted intensity and s the scale. "
                    "'ev11' (default): a*sigma^2 + b^2*p^2. 'photon': adds c*p, a counting term linear in the raw "
                    "intensity. 'gscaled': adds c*s*p, Aimless's SdB term (linear in the scaled intensity). "
                    "'aimless': Aimless's SDcorrection as written, SdFac^2*max(0.1*sigma^2, sigma^2 + SdB*s*p + "
                    "SdAdd*|SdAdd|*p^2), with SdFac >= 0.2, |SdAdd| <= 1 and SdB and SdAdd free in sign. "
                    "c, SdFac, SdB and SdAdd are learned on the ELBO (--error-model-fit elbo or normal).",
            "default": "ev11",
            "choices": ["ev11", "photon", "gscaled", "aimless"],
        }
    ),
    (
        (
            "--error-model-burnin",
        ),{
            "help": "--error-model-fit online: epochs to hold a and b at their starting values before the fit starts "
                    "accumulating. Default 2.",
            "default": 2,
            "type": int,
        }
    ),
    (
        (
            "--error-model-fit-a-only",
        ),{
            "help": "--error-model ev11: fit a alone and hold b at --error-model-b (either --error-model-fit).",
            "action": "store_true",
        }
    ),
    (
        (
            "--error-model-learn-dof",
        ),{
            "help": "--error-model ev11 with --error-model-fit elbo: also learn the Student t's degrees of freedom, "
                    "starting from --studentt-dof (held with a and b by --fix-error-model).",
            "action": "store_true",
        }
    ),
    (
        (
            "--error-model-decay",
        ),{
            "help": "--error-model-fit online: forget the running sums with this time constant in steps. The default, "
                    "0, keeps every batch since the burn-in (a cumulative fit).",
            "default": 0,
            "type": int,
        }
    ),
    (
        (
            "--background",
        ),{
            "help": "An additive background term, I_pred = s*T*F^2 + B, from extra outputs of the scale network. "
                    "'offset': B = sigma*beta, beta a point estimate with a N(0, --background-prior-width^2) penalty. "
                    "'latent': B ~ N(sigbg*m, sigbg*s) with KL to N(0, sigbg^2), sigbg the integration program's "
                    "background uncertainty from raw metadata column --background-sigma-column. 'none' (default).",
            "default": "none",
            "choices": ["none", "offset", "latent"],
        }
    ),
    (
        (
            "--background-prior-width",
        ),{
            "help": "--background offset: the prior width of B/sigma, default 1.",
            "default": 1.0,
            "type": float,
        }
    ),
    (
        (
            "--background-sigma-column",
        ),{
            "help": "--background latent: the metadata column (0-based, as the loader orders them) holding the "
                    "background uncertainty in intensity units: 4 for DIALS (sqrt(background.sum.variance)), 1 for "
                    "the default MTZ columns (SIGBG).",
            "default": None,
            "type": int,
        }
    ),
)
