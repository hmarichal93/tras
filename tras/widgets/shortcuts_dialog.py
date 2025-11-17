from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt


class ShortcutsDialog(QtWidgets.QDialog):
    """Dialog to display all keyboard shortcuts."""
    
    def __init__(self, shortcuts_config, parent=None):
        super().__init__(parent)
        self.shortcuts_config = shortcuts_config
        self.setWindowTitle("Keyboard Shortcuts")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QtWidgets.QVBoxLayout()
        
        # Title
        title = QtWidgets.QLabel("<h2>⌨️ Keyboard Shortcuts</h2>")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Create scrollable area
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Container widget for all shortcuts
        container = QtWidgets.QWidget()
        container_layout = QtWidgets.QVBoxLayout()
        
        # Group shortcuts by category
        categories = {
            "File Operations": [
                ("open", "Open Image/Label File"),
                ("open_dir", "Open Directory"),
                ("open_next", "Next Image"),
                ("open_prev", "Previous Image"),
                ("save", "Save"),
                ("save_as", "Save As"),
                ("close", "Close File"),
                ("delete_file", "Delete File"),
                ("quit", "Quit Application"),
            ],
            "TRAS Workflow": [
                ("sample_metadata", "Sample Metadata"),
                ("set_scale", "Set Image Scale"),
                ("preprocess_image", "Preprocess Image"),
                ("detect_tree_rings", "Detect Tree Rings"),
                ("measure_radial_width", "Measure Ring Width"),
                ("view_ring_properties", "View Ring Properties"),
            ],
            "Drawing & Editing": [
                ("create_polygon", "Create Ring"),
                ("create_rectangle", "Create Rectangle"),
                ("edit_polygon", "Edit Mode"),
                ("edit_label", "Edit Label"),
                ("delete_polygon", "Delete Ring"),
                ("duplicate_polygon", "Duplicate Ring"),
                ("copy_polygon", "Copy Ring"),
                ("paste_polygon", "Paste Ring"),
                ("undo", "Undo"),
                ("remove_selected_point", "Remove Selected Point"),
                (
                    "manual_add_point_edge",
                    "Add Point to Existing Ring",
                    "Alt + Click edge while in Edit mode",
                ),
            ],
            "View & Zoom": [
                ("zoom_in", "Zoom In"),
                ("zoom_out", "Zoom Out"),
                ("zoom_to_original", "Zoom to 100%"),
                ("fit_window", "Fit to Window"),
                ("fit_width", "Fit Width"),
                ("toggle_all_polygons", "Toggle All Rings Visibility"),
            ],
        }
        
        for category_name, shortcuts_list in categories.items():
            # Category header
            header = QtWidgets.QLabel(f"<h3>{category_name}</h3>")
            header.setStyleSheet("color: #8B5A2B; margin-top: 10px;")
            container_layout.addWidget(header)
            
            # Create table for this category
            table = QtWidgets.QTableWidget()
            table.setColumnCount(2)
            table.setHorizontalHeaderLabels(["Action", "Shortcut"])
            table.horizontalHeader().setStretchLastSection(True)
            table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
            table.verticalHeader().setVisible(False)
            table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
            table.setAlternatingRowColors(True)
            
            # Populate table
            valid_shortcuts = []
            for shortcut_entry in shortcuts_list:
                if len(shortcut_entry) == 2:
                    key, description = shortcut_entry
                    shortcut_value = self.shortcuts_config.get(key)
                else:
                    key, description, shortcut_value = shortcut_entry
                if shortcut_value:
                    valid_shortcuts.append((description, shortcut_value))
            
            table.setRowCount(len(valid_shortcuts))
            for row, (description, shortcut_value) in enumerate(valid_shortcuts):
                # Action column
                action_item = QtWidgets.QTableWidgetItem(description)
                table.setItem(row, 0, action_item)
                
                # Shortcut column - format nicely
                if isinstance(shortcut_value, list):
                    shortcut_text = " or ".join([self._format_shortcut(s) for s in shortcut_value])
                else:
                    shortcut_text = self._format_shortcut(shortcut_value)
                
                shortcut_item = QtWidgets.QTableWidgetItem(shortcut_text)
                shortcut_item.setTextAlignment(Qt.AlignCenter)
                table.setItem(row, 1, shortcut_item)
            
            table.resizeRowsToContents()
            container_layout.addWidget(table)
        
        # Add note at bottom
        note = QtWidgets.QLabel(
            "<p style='color: gray; font-style: italic; margin-top: 20px;'>"
            "💡 Tip: You can customize shortcuts by editing ~/.trasrc"
            "</p>"
        )
        note.setWordWrap(True)
        container_layout.addWidget(note)
        
        container_layout.addStretch()
        container.setLayout(container_layout)
        scroll.setWidget(container)
        
        layout.addWidget(scroll)
        
        # Close button
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_btn.setMinimumWidth(100)
        button_layout.addWidget(close_btn)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def _format_shortcut(self, shortcut):
        """Format shortcut string for display."""
        if not shortcut:
            return "—"
        # Replace common key names for better readability
        formatted = str(shortcut)
        formatted = formatted.replace("Ctrl", "Ctrl")
        formatted = formatted.replace("Shift", "Shift")
        formatted = formatted.replace("Alt", "Alt")
        formatted = formatted.replace("Meta", "Meta")
        return formatted
