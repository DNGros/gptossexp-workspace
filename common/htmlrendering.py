"""
HTML table rendering utilities.
Generates HTML tables from nested dictionary structures, parallel to LaTeX table generation.
"""

import datetime
import subprocess
from workspace.common.texrendering import (
    _analyze_column_structure,
    _get_flat_headers,
    _get_max_depth,
    _get_item_depth,
    _flatten_row_values,
    _collect_leaf_headers,
    get_git_info
)


def escape_html(text):
    """Escape HTML special characters and convert LaTeX math to HTML entities."""
    if not isinstance(text, str):
        text = str(text)
    
    # First, convert common LaTeX math symbols to HTML entities
    text = text.replace('$\\pm$', '±')
    text = text.replace('\\pm', '±')
    
    # Then escape HTML special characters
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')
    text = text.replace('"', '&quot;')
    text = text.replace("'", '&#x27;')
    
    return text


def generate_html_table(rows, table_class=""):
    """Generate an HTML table from nested dict structure.
    
    Args:
        rows: List of dictionaries containing table data, or strings for special row markers
        table_class: Optional CSS class(es) to add to the table element
        
    Returns:
        HTML string containing just the <table> element (no figure/caption wrapper)
    """
    
    # Find the first dictionary row to analyze structure
    sample_row = None
    for row in rows:
        if isinstance(row, dict):
            sample_row = row
            break
    
    if sample_row is None:
        raise ValueError("No dictionary rows found in data")
    
    # Analyze the column structure recursively (reuse from texrendering)
    column_structure = _analyze_column_structure(sample_row)
    max_depth = _get_max_depth(column_structure)
    
    html = []
    
    # Add metadata comment
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    git_sha = get_git_info()
    html.append(f"<!-- Generated on {timestamp} from git commit {git_sha} -->")
    html.append("")
    
    # Table element
    table_classes = f' class="{table_class}"' if table_class else ''
    html.append(f'<table{table_classes}>')
    
    # Generate header
    html.append('  <thead>')
    _generate_html_headers(html, column_structure, max_depth)
    html.append('  </thead>')
    
    # Generate body
    html.append('  <tbody>')
    for row in rows:
        if isinstance(row, str):
            # Skip string rows (they're for LaTeX formatting like \midrule)
            continue
        elif isinstance(row, dict):
            values = _flatten_row_values(row)
            html.append('    <tr>')
            for v in values:
                html.append(f'      <td>{escape_html(str(v))}</td>')
            html.append('    </tr>')
    html.append('  </tbody>')
    
    html.append('</table>')
    
    return '\n'.join(html)


def _generate_html_headers(html, structure, max_depth):
    """Generate HTML table headers with proper colspan for nested columns."""
    
    if max_depth == 1:
        # Simple case: single header row
        html.append('    <tr>')
        for item in structure:
            html.append(f'      <th>{escape_html(item["name"])}</th>')
        html.append('    </tr>')
        return
    
    # Generate multi-level headers
    for level in range(max_depth):
        html.append('    <tr>')
        header_row = _build_html_header_row_at_level(structure, level, max_depth)
        for cell in header_row:
            html.append(f'      {cell}')
        html.append('    </tr>')


def _build_html_header_row_at_level(structure, target_level, max_depth):
    """Build header cells for a specific level."""
    cells = []
    
    for item in structure:
        item_depth = _get_item_depth(item)
        
        if target_level == 0:
            # Top level
            if item['type'] == 'group' and item_depth > 1:
                # Add class for group headers (will have bottom border)
                cells.append(f'<th colspan="{item["span"]}" class="group-header">{escape_html(item["name"])}</th>')
            elif item['type'] == 'leaf':
                cells.append('<th></th>')  # Empty for single columns at top level
            else:
                cells.append('<th></th>')
        
        elif target_level == max_depth - 1:
            # Bottom level - actual column names
            if item['type'] == 'leaf':
                cells.append(f'<th>{escape_html(item["name"])}</th>')
            else:
                # For groups, collect leaf headers
                leaf_headers = []
                _collect_leaf_headers_html(item, leaf_headers)
                cells.extend([f'<th>{h}</th>' for h in leaf_headers])
        
        else:
            # Middle levels - handle nested groups
            middle_cells = _get_html_headers_at_middle_level(item, target_level, max_depth)
            cells.extend(middle_cells)
    
    return cells


def _get_html_headers_at_middle_level(item, target_level, max_depth):
    """Get header cells for middle levels of nesting."""
    cells = []
    
    if item['type'] == 'leaf':
        cells.append('<th></th>')
        return cells
    
    def process_level(structure, remaining_depth):
        level_cells = []
        
        if remaining_depth == 1:
            for child in structure:
                if child['type'] == 'group':
                    # Add class for group headers (will have bottom border)
                    level_cells.append(f'<th colspan="{child["span"]}" class="group-header">{escape_html(child["name"])}</th>')
                else:
                    level_cells.append('<th></th>')
        else:
            for child in structure:
                if child['type'] == 'group':
                    child_cells = process_level(child['children'], remaining_depth - 1)
                    level_cells.extend(child_cells)
                else:
                    level_cells.append('<th></th>')
        
        return level_cells
    
    if item['type'] == 'group':
        return process_level(item['children'], target_level)
    
    return cells


def _collect_leaf_headers_html(item, headers):
    """Collect all leaf headers from an item (HTML version)."""
    if item['type'] == 'leaf':
        headers.append(escape_html(item['name']))
    else:
        for child in item['children']:
            _collect_leaf_headers_html(child, headers)

