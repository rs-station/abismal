"""Bridging the two ipywidgets majors the GUI has to run under.

Colab is the reason. Its notebooks ship ipywidgets 7.7.1, and its widget manager
only knows the models that generation emits. Upgrading to 8 there does not work:
Colab's opt-in CDN manager is pinned to an ipywidgets-8 *alpha* (html-manager
5.0.0a) that has `ButtonStyleModel` but not `HTMLStyleModel` or `TextStyleModel`,
which 8.1.9 attaches to every HTML and Text widget. The console says

    Cannot find model module @jupyter-widgets/controls@2.0.0, HTMLStyleModel

and the widget renders as nothing at all -- no output, no Python error. A form
that is 108 Text and 102 HTML widgets renders as a blank cell.

On 7 every widget carries `DescriptionStyleModel` or `ButtonStyleModel`, both of
which Colab has always known, so staying on 7 there is what actually works -- and
it costs nothing elsewhere, since 8 is a superset for everything this form uses.
"""
import ipywidgets as widgets


def set_tooltip(widget, text):
    """Attach hover help, under whichever name this ipywidgets spells it.

    ipywidgets 8 gives every widget a `tooltip`. 7 has it on Button only, and
    spells it `description_tooltip` on everything with a description. Passing the
    wrong one to a 7.x widget is not an error -- traitlets warns about an
    unrecognised argument and drops the value -- so tooltips simply vanish, which
    is how they went missing on dropdowns before.

    Asking the widget rather than the version number keeps this right for widgets
    that follow their own rules, Button being the one that always has.
    """
    # Empty text is set, not skipped. "this argument has no help" is a real
    # state, and leaving the trait at its default None would make it
    # indistinguishable from a control that was never given a tooltip at all --
    # which is the difference the form's tests turn on.
    if widget.has_trait("tooltip"):
        widget.tooltip = text
    elif widget.has_trait("description_tooltip"):
        widget.description_tooltip = text
    return widget


def supports_style_models():
    """Whether this ipywidgets emits the per-widget style models 8 introduced.

    True on 8, False on 7. Only interesting for deciding whether Colab's built-in
    widget manager can render what we are about to hand it.
    """
    return int(widgets.__version__.split(".")[0]) >= 8
