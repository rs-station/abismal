from traitlets import List,Unicode
from ipywidgets import widgets
import solara

class ServerFileSelectorWidget(widgets.VBox):
    """
    An ipywidget that wraps Solara's FileBrowser for server-side file selection.
    Allows selection of multiple files and exposes selected paths as a traitlet.
    """
    
    # Traitlets
    selected_files = List(Unicode(), default_value=[]).tag(sync=True)
    current_directory = Unicode(default_value=".").tag(sync=True)
    header_string = "Select Files"

    def __init__(self, initial_directory=".", **kwargs):
        """
        Initialize the ServerFileSelectorWidget.
        
        Parameters
        ----------
        initial_directory : str, optional
            Starting directory for the file browser (default: ".")
        file_filter : callable, optional
            Function to filter files (takes path, returns bool)
        **kwargs
            Additional arguments passed to VBox
        """
        self.current_directory = initial_directory

        # Create UI components
        self._create_widgets()

        # Initialize VBox with children
        super().__init__(
            children=[
                self.header_label,
                self.file_browser_output,
                self.selected_files_label,
                self.button_box
            ],
            **kwargs
        )
        
        # Render Solara component
        self._render_file_browser()
    
    def file_filter(self, file_name):
        return True

    def _create_widgets(self):
        """Create the widget components."""
        # Header
        self.header_label = widgets.HTML(
            value=f"<h3>{self.header_string}</h3>"
        )
        
        # Output widget to render Solara component
        self.file_browser_output = widgets.Output()
        
        # Display selected files
        self.selected_files_label = widgets.HTML(
            value=self._format_selected_files()
        )
        
        # Buttons
        self.clear_button = widgets.Button(
            description="Clear Selection",
            button_style="warning",
            icon="times"
        )
        self.clear_button.on_click(self._on_clear_clicked)

        self.button_box = widgets.HBox([
            self.clear_button,
        ])

    def _render_file_browser(self):
        """Render the Solara FileBrowser component."""
        with self.file_browser_output:
            self.file_browser_output.clear_output(wait=True)
            
            # Create reactive variables for Solara
            self._selected_paths = solara.reactive([])

            @solara.component
            def FileBrowserComponent():
                """Solara component wrapper for FileBrowser."""
                #from solara.components import FileBrowser
                from abismal.gui.components.file_browser import FileBrowser
                #from solara.lab import FileBrowser
                import os

                def on_path_select(path):
                    pass

                def on_directory_change(path):
                    """Handle changing working directory"""
                    name = str(path)
                    self.current_directory = name

                def on_file_select(path):
                    """Handle file selection."""
                    name = str(path)
                    current = list(self._selected_paths.value)
                    if not self.file_filter(name):
                        return
                    if path in current:
                        current.remove(name)
                    else:
                        current.append(name)
                    self._selected_paths.value = current

                    # Update traitlet
                    self.selected_files = list(self._selected_paths.value)
                    self._update_selected_files_display()

                FileBrowser(
                    directory=self.current_directory,
                    on_file_open=on_file_select,
                    on_directory_change=on_directory_change,
                    on_path_select=on_path_select,
                    can_select=True,
                )

            # Display the Solara component
            display(FileBrowserComponent())
    
    def _format_selected_files(self):
        """Format the selected files list for display."""
        if not self.selected_files:
            return "<p><i>No files selected</i></p>"
        
        files_html = "<p><b>Selected files:</b></p><ul>"
        for file_path in self.selected_files:
            files_html += f"<li><code>{file_path}</code></li>"
        files_html += "</ul>"
        
        return files_html
    
    def _update_selected_files_display(self):
        """Update the display of selected files."""
        self.selected_files_label.value = self._format_selected_files()
    
    def _on_clear_clicked(self, button):
        """Handle clear button click."""
        self.selected_files = []
        self._selected_paths.value = []
        self._update_selected_files_display()
    
    def get_selected_files(self):
        """
        Get the list of selected file paths.
        
        Returns
        -------
        list of str
            List of selected file paths
        """
        return list(self.selected_files)
    
    def set_directory(self, directory):
        """
        Change the current directory.
        
        Parameters
        ----------
        directory : str
            New directory path
        """
        self.current_directory = directory
        self._render_file_browser()

    @property
    def value(self):
        return ' '.join(self.get_selected_files())

class ReflectionFileSelector(ServerFileSelectorWidget):
    header_string = "Input Reflection Files (*.stream, *.expt/refl, *.mtz)"
    file_types = [
        '.mtz',
        '.expt', '.refl',
        '.json', '.pickle', #Secretly support legacy dials formats for now
        '.stream',
    ]
    def __init__(self, *args, **kwargs):
        super().__init__()

    def file_filter(self, file_name):
        out = False
        for suffix in self.file_types:
            out |= file_name.endswith(suffix)
        return out

class PhenixFileSelector(ReflectionFileSelector):
    header_string = "Configuration (*.eff) file for phenix.refine"
    file_types = [
        '.eff',
    ]
    def value(self):
        return ','.join(self.get_selected_files())


