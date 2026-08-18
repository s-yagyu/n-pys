import os
import textwrap
import base64
from io import BytesIO
from typing import List, Optional, Dict

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader

# Font setting (adjust based on your environment if specific characters are needed)
plt.rcParams['font.family'] = 'sans-serif'

class ReportBuilder:
    """Integrated module for exporting reports in PDF, PNG, or SVG formats."""

    def __init__(self, dpi: int = 150):
        self.store: Dict[str, BytesIO] = {}
        self.dpi = dpi
        
        # Determine drawing width based on A4 page size
        self.page_width, self.page_height = A4
        self.margin = 0.5 * inch
        self.content_width_inch = (self.page_width - 2 * self.margin) / inch

    def _figure_to_buffer(self, fig: plt.Figure) -> BytesIO:
        """Convert a Matplotlib Figure to a PNG buffer and save it."""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        return buf

    def add_graph(self, name: str, fig: plt.Figure, show: bool = False, save_svg_path: Optional[str] = None):
        """Add a graph, optionally display it or save an individual SVG."""
        if show:
            plt.show()

        if save_svg_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_svg_path)), exist_ok=True)
            fig.savefig(save_svg_path, format='svg', bbox_inches='tight')
            print(f"Vector SVG saved to: {save_svg_path}")

        self.store[name] = self._figure_to_buffer(fig)
        plt.close(fig)

    def add_text(self, name: str, text: str, title: Optional[str] = None, fontsize: int = 12):
        """Add a text block with automatic word wrapping."""
        char_per_line = int(self.content_width_inch * 10 * (12/fontsize))
        wrapper = textwrap.TextWrapper(width=char_per_line)
        
        # lines = wrapper.wrap(text)
        lines = []
        for p in text.split('\n'):
            if p == '':
                lines.append('')
            else:
                lines.extend(wrapper.wrap(p))
        
        num_lines = len(lines)
        
        if title:
            num_lines += 2
       
        line_height_inch = (fontsize * 1.5) / 72
        fig_height = max(0.5, num_lines * line_height_inch + 0.2)

        fig, ax = plt.subplots(figsize=(self.content_width_inch, fig_height))
        ax.axis('off')
        
        current_y = 1.0
        if title:
            ax.text(0.0, current_y, title, fontsize=fontsize+2, fontweight='bold', 
                    va='top', ha='left', transform=ax.transAxes)
            current_y -= (line_height_inch * 2) / fig_height

        wrapped_text = "\n".join(lines)
        ax.text(0.0, current_y, wrapped_text, fontsize=fontsize, 
                va='top', ha='left', transform=ax.transAxes)

        self.store[name] = self._figure_to_buffer(fig)
        plt.close(fig)

    def add_dataframe(self, name: str, df: pd.DataFrame, 
                      max_chars: int = 20, max_cols: int = 10, 
                      fontsize: Optional[int] = None,
                      header_fontsize: Optional[int] = None,
                      header_weight: Optional[str] = None):
        """Add a DataFrame as a table with advanced auto-adjustment."""
        if fontsize is None:
            if max_cols <= 5: calc_size = 10
            elif max_cols <= 8: calc_size = 9
            elif max_cols <= 12: calc_size = 8
            elif max_cols <= 20: calc_size = 7
            else: calc_size = 6
            fontsize = calc_size
            
        if header_weight is None:
            header_weight = 'normal' if fontsize < 8 else 'bold'
            
        if header_fontsize is None:
            header_fontsize = fontsize

        all_columns = df.columns
        col_chunks = [all_columns[i:i + max_cols] for i in range(0, len(all_columns), max_cols)]

        table_blocks = []
        total_fig_height = 0.0
        gap_height = 0.3
        line_height_factor = (fontsize * 1.8) / 72  
        wrapper = textwrap.TextWrapper(width=max_chars, break_long_words=True)

        for chunk_cols in col_chunks:
            sub_df = df[chunk_cols].copy().astype(str)
            new_columns = ["\n".join(wrapper.wrap(str(c))) for c in sub_df.columns]
            sub_df.columns = new_columns
            for col in sub_df.columns:
                sub_df[col] = sub_df[col].apply(lambda x: "\n".join(wrapper.wrap(str(x))))

            row_max_lines = []
            for _, row in sub_df.iterrows():
                lines_in_row = [str(cell).count('\n') + 1 for cell in row]
                row_max_lines.append(max(lines_in_row) if lines_in_row else 1)
            
            header_lines = max([c.count('\n') + 1 for c in sub_df.columns]) if len(sub_df.columns) > 0 else 1
            total_lines_in_block = sum(row_max_lines) + header_lines
            block_height = max(0.2, total_lines_in_block * line_height_factor)
            
            table_blocks.append({
                "df": sub_df, "height": block_height,
                "row_lines": row_max_lines, "header_lines": header_lines
            })
            total_fig_height += block_height + gap_height

        if total_fig_height > gap_height:
            total_fig_height -= gap_height

        fig, ax = plt.subplots(figsize=(self.content_width_inch, total_fig_height))
        ax.axis('off')
        current_y = 1.0

        for block in table_blocks:
            sub_df = block["df"]
            h_inch = block["height"]
            rel_height = h_inch / total_fig_height
            bbox = [0.0, current_y - rel_height, 1.0, rel_height]
            
            table = ax.table(cellText=sub_df.values, colLabels=sub_df.columns,
                             bbox=bbox, loc='center', cellLoc='left')
            table.auto_set_font_size(False)
            table.set_fontsize(fontsize)
            
            if len(sub_df.columns) > 0:
                table.auto_set_column_width(col=list(range(len(sub_df.columns))))

            cells = table.get_celld()
            row_lines = block["row_lines"]
            header_lines = block["header_lines"]
            unit_height = (line_height_factor / h_inch) * rel_height

            for col_idx in range(len(sub_df.columns)):
                cell = cells[(0, col_idx)]
                cell.set_height(unit_height * header_lines)
                cell.set_facecolor('#f0f0f0')
                cell.set_text_props(weight=header_weight, ha='center')
                cell.set_fontsize(header_fontsize)
                cell.set_linewidth(0.5)

            current_row_idx = 1
            for lines_count in row_lines:
                row_h = unit_height * lines_count
                for col_idx in range(len(sub_df.columns)):
                    cell = cells[(current_row_idx, col_idx)]
                    cell.set_height(row_h)
                    cell.set_linewidth(0.5)
                current_row_idx += 1

            rel_gap = gap_height / total_fig_height
            current_y -= (rel_height + rel_gap)

        self.store[name] = self._figure_to_buffer(fig)
        plt.close(fig)

    def generate_report(self, order_list: List[str], title: str = "Report", 
                        filename: Optional[str] = None, report_format: str = 'pdf', gap: int = 15):
        """Generate a report in PDF, PNG, or SVG format."""
        report_format = report_format.lower()
        if filename is None:
            filename = f"{title}.{report_format}"

        if report_format == 'pdf':
            self._generate_pdf(order_list, title, filename, gap)
        elif report_format == 'png':
            self._generate_png(order_list, filename, gap)
        elif report_format == 'svg':
            self._generate_svg(order_list, filename, gap)
        else:
            print(f"Error: Unsupported format '{report_format}'")

    def _generate_pdf(self, order_list, title, filename, gap):
        """PDF generation logic."""
        c = canvas.Canvas(filename, pagesize=A4)
        y_position = self.page_height - self.margin
        
        c.setFont('Helvetica-Bold', 16)
        c.drawString(self.margin, y_position - 16, title)
        y_position -= 50
       
        for name in order_list:
            if name not in self.store:
                continue
            self.store[name].seek(0)
            img = Image.open(self.store[name])
            if img.width == 0 or img.height == 0: continue
            
            target_width = self.page_width - 2 * self.margin
            target_height = target_width * (img.height / img.width)
            
            # --- Added: if an image is too tall, scale it down while preserving aspect ratio ---
            max_height_limit = self.page_height * 0.55  # Limit height to 55% of A4 portrait
            if target_height > max_height_limit:
                scale_ratio = max_height_limit / target_height
                target_height = max_height_limit
                target_width = target_width * scale_ratio
                
            # X offset for centering the image
            offset_x = self.margin + ((self.page_width - 2 * self.margin) - target_width) / 2
            # ---------------------------------------------------------------------
     
            if y_position - target_height < self.margin:
                c.showPage()
                y_position = self.page_height - self.margin
                c.setFont('Helvetica-Bold', 10)
                c.drawString(self.margin, y_position, f"{title} (cont.)")
                y_position -= 30
            
            c.drawImage(ImageReader(img), offset_x, y_position - target_height,
                        width=target_width, height=target_height, mask='auto')
            y_position -= (target_height + gap)
            
        c.save()
        print(f"PDF Report generated: {filename}")

    def _generate_png(self, order_list, filename, gap):
        """PNG vertical concatenation logic."""
        images = []
        max_width, total_height = 0, 0
        for name in order_list:
            if name in self.store:
                self.store[name].seek(0)
                img = Image.open(self.store[name])
                images.append(img)
                max_width = max(max_width, img.width)
                total_height += img.height + gap

        if not images: return
        combined_img = Image.new('RGB', (max_width, total_height), (255, 255, 255))
        current_y = 0
        for img in images:
            combined_img.paste(img, (0, current_y))
            current_y += img.height + gap

        combined_img.save(filename)
        print(f"PNG Report generated: {filename}")

    def _generate_svg(self, order_list, filename, gap):
        """SVG report generation logic (embeds images using Base64 encoding)."""
        images_data = []
        max_width, total_height = 0, 0
        
        for name in order_list:
            if name in self.store:
                self.store[name].seek(0)
                img_bytes = self.store[name].read()
                img = Image.open(BytesIO(img_bytes))
                
                b64_data = base64.b64encode(img_bytes).decode('utf-8')
                images_data.append({
                    'data': b64_data,
                    'width': img.width,
                    'height': img.height
                })
                max_width = max(max_width, img.width)
                total_height += img.height + gap

        if not images_data: return

        # Assemble SVG content
        svg_content = [
            f'<svg width="{max_width}" height="{total_height}" xmlns="http://www.w3.org/2000/svg">',
            f'<rect width="100%" height="100%" fill="white"/>'
        ]
        
        current_y = 0
        for item in images_data:
            img_tag = f'<image href="data:image/png;base64,{item["data"]}" x="0" y="{current_y}" width="{item["width"]}" height="{item["height"]}"/>'
            svg_content.append(img_tag)
            current_y += item["height"] + gap
            
        svg_content.append('</svg>')

        with open(filename, 'w', encoding='utf-8') as f:
            f.write("\n".join(svg_content))
            
        print(f"SVG Report generated: {filename}")


# ==========================================
# Test code and usage examples
# ==========================================
if __name__ == "__main__":
    # ---------------------------------------------------------
    # Usage example (commented out):
    # Example of how to import and use this module within an existing project.
    # ---------------------------------------------------------
    """
    from integrated_report import ReportBuilder
    import pandas as pd
    import matplotlib.pyplot as plt

    manager = ReportBuilder(dpi=200)

    # 1. Add text
    manager.add_text("intro", "This report summarizes the analysis results of sales data.", title="Monthly Report")

    # 2. Add a graph
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [10, 20, 15])
    ax.set_title("Sales Trend")
    # Specify save_svg_path if you also want to save the individual graph as a vector SVG
    manager.add_graph("trend_chart", fig, save_svg_path="output/trend.svg")

    # 3. Add a DataFrame (table)
    df = pd.DataFrame({'Product': ['A', 'B'], 'Sales': [100, 200]})
    manager.add_dataframe("sales_table", df)

    # 4. Generate the report (choose from 'pdf', 'png', or 'svg')
    order = ["intro", "trend_chart", "sales_table"]
    manager.generate_report(order, title="Monthly Report", report_format="pdf")
    manager.generate_report(order, title="Monthly Report", report_format="svg")
    """

    # ---------------------------------------------------------
    # Test execution code for verification
    # ---------------------------------------------------------
    print("--- Starting integration module test ---")
    manager = ReportBuilder()

    # Create sample data
    cols = [f"Col_{i}" for i in range(8)]
    data = [[round(np.random.rand(), 2) for _ in range(8)] for _ in range(3)]
    df_sample = pd.DataFrame(data, columns=cols)
    
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(['A', 'B', 'C'], [3, 7, 2], color='skyblue')
    ax.set_title("Sample Bar Chart")

    # Register content with the module
    manager.add_text("t1", "This is an integration test for the report generation module. It outputs to PDF, PNG, and SVG formats.", title="Integration Test")
    manager.add_graph("g1", fig)
    manager.add_dataframe("d1", df_sample, max_cols=5)

    # Output test for all formats
    output_items = ["t1", "g1", "d1"]
    manager.generate_report(output_items, title="Test Report", report_format="pdf", filename="test_output.pdf")
    manager.generate_report(output_items, title="Test Report", report_format="png", filename="test_output.png")
    manager.generate_report(output_items, title="Test Report", report_format="svg", filename="test_output.svg")
    print("--- Test completed ---")