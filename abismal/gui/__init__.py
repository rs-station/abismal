from .components import ArgparseGUI, GemmiMolViewer
# Private module: `abismal.gui.launch` should unambiguously be the
# function, not a submodule shadowed by it.
from ._launch import launch


def enable_colab_widget_manager():
    """Point Colab at a widget manager that can render ipywidgets 8.

    Colab's built-in manager is the ipywidgets 7 generation, and abismal needs 8
    -- the `tooltip` trait that 7.x spells description_tooltip, on a dozen
    controls. Handed v8 widgets, the built-in manager renders *nothing*: the cell
    succeeds, the output area is empty, and no error appears anywhere. That is
    the hardest shape of failure to diagnose, and it is what happens without this
    call.

    Colab ships an opt-in manager loaded from its CDN (html-manager 5.x, the v8
    generation) for exactly this. It is what the "Third-party Jupyter widgets"
    snippet in Colab's own snippet panel pastes in; doing it here means nobody
    has to know that.

    A no-op anywhere else, and best-effort on Colab: if it fails, the widgets may
    not render, but raising here would take down an import that is otherwise fine.
    Returns whether the manager was switched, so callers and tests can tell.
    """
    from abismal.gui.components._compat import supports_style_models

    # Only under ipywidgets 8. On 7 Colab's built-in manager already renders
    # everything we emit, and switching to the CDN one -- which expects controls
    # 2.0 -- would break what currently works.
    if not supports_style_models():
        return False

    try:
        from google.colab import output
    except ImportError:
        return False
    try:
        output.enable_custom_widget_manager()
        return True
    except Exception:
        return False


# At import, because it has to happen before any widget is displayed and the
# alternative is boilerplate in every notebook that forgets it once.
enable_colab_widget_manager()
