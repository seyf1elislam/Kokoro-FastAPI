import gradio as gr
import pytest

from ui.lib.components.input import create_input_column


def test_create_input_column_structure():
    """Test that create_input_column returns the expected structure"""
    column, components = create_input_column()

    # Test the return types
    assert isinstance(column, gr.Column)
    assert isinstance(components, dict)

    # Test that all expected components are present
    expected_components = {
        "tabs",
        "text_input",
        "file_select",
        "file_upload",
        "file_preview",
        "text_submit",
        "file_submit",
        "clear_files",
    }
    assert set(components.keys()) == expected_components

    # Test component types
    assert isinstance(components["tabs"], gr.Tabs)
    assert isinstance(components["text_input"], gr.Textbox)
    assert isinstance(components["file_select"], gr.Dropdown)
    assert isinstance(components["file_upload"], gr.File)
    assert isinstance(components["file_preview"], gr.Textbox)
    assert isinstance(components["text_submit"], gr.Button)
    assert isinstance(components["file_submit"], gr.Button)
    assert isinstance(components["clear_files"], gr.Button)


def test_text_input_configuration():
    """Test the text input component configuration"""
    _, components = create_input_column()
    text_input = components["text_input"]

    assert text_input.label == "Text to speak"
    assert text_input.placeholder == "Enter text here..."
    assert text_input.lines == 4


def test_file_upload_configuration():
    """Test the file upload component configuration"""
    _, components = create_input_column()
    file_upload = components["file_upload"]

    assert file_upload.label == "Upload Text File (.txt)"
    assert file_upload.file_types == [".txt"]


def test_button_configurations():
    """Test the button configurations"""
    _, components = create_input_column()

    # Test text submit button
    assert components["text_submit"].value == "Generate Speech"
    assert components["text_submit"].variant == "primary"
    assert components["text_submit"].size == "lg"

    # Test file submit button
    assert components["file_submit"].value == "Generate Speech"
    assert components["file_submit"].variant == "primary"
    assert components["file_submit"].size == "lg"

    # Test clear files button
    assert components["clear_files"].value == "Clear Files"
    assert components["clear_files"].variant == "secondary"
    assert components["clear_files"].size == "lg"
