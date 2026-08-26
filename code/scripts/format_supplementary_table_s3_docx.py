"""Apply a readable landscape layout to the proposed Table S3 DOCX."""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

from lxml import etree


WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": WORD_NS}


def w_tag(name: str) -> str:
    return f"{{{WORD_NS}}}{name}"


def set_attr(element: etree._Element, name: str, value: str) -> None:
    element.set(w_tag(name), value)


def get_or_add(parent: etree._Element, name: str) -> etree._Element:
    child = parent.find(f"w:{name}", NS)
    if child is None:
        child = etree.SubElement(parent, w_tag(name))
    return child


def get_or_prepend(parent: etree._Element, name: str) -> etree._Element:
    child = parent.find(f"w:{name}", NS)
    if child is None:
        child = etree.Element(w_tag(name))
        parent.insert(0, child)
    return child


def set_page_layout(root: etree._Element) -> None:
    section = root.find(".//w:sectPr", NS)
    if section is None:
        raise ValueError("DOCX has no section properties")
    page_size = get_or_add(section, "pgSz")
    set_attr(page_size, "w", "15840")
    set_attr(page_size, "h", "12240")
    set_attr(page_size, "orient", "landscape")
    margins = get_or_add(section, "pgMar")
    for name, value in {
        "top": "720",
        "right": "720",
        "bottom": "720",
        "left": "720",
        "header": "360",
        "footer": "360",
        "gutter": "0",
    }.items():
        set_attr(margins, name, value)


def set_run_size(run: etree._Element, half_points: int) -> None:
    run_properties = run.find("w:rPr", NS)
    if run_properties is None:
        run_properties = etree.Element(w_tag("rPr"))
        run.insert(0, run_properties)
    for name in ("sz", "szCs"):
        size = get_or_add(run_properties, name)
        set_attr(size, "val", str(half_points))


def set_cell_width(cell: etree._Element, width: int) -> None:
    properties = get_or_prepend(cell, "tcPr")
    cell_width = get_or_add(properties, "tcW")
    set_attr(cell_width, "w", str(width))
    set_attr(cell_width, "type", "dxa")
    vertical_align = get_or_add(properties, "vAlign")
    set_attr(vertical_align, "val", "top")


def set_table_layout(table: etree._Element, widths: list[int], font_size: int) -> None:
    properties = get_or_add(table, "tblPr")
    table_width = get_or_add(properties, "tblW")
    set_attr(table_width, "w", str(sum(widths)))
    set_attr(table_width, "type", "dxa")
    layout = get_or_add(properties, "tblLayout")
    set_attr(layout, "type", "fixed")

    grid = table.find("w:tblGrid", NS)
    if grid is None:
        raise ValueError("Table has no grid")
    columns = grid.findall("w:gridCol", NS)
    if len(columns) != len(widths):
        raise ValueError("Unexpected table column count")
    for column, width in zip(columns, widths, strict=True):
        set_attr(column, "w", str(width))

    rows = table.findall("w:tr", NS)
    if rows:
        row_properties = get_or_prepend(rows[0], "trPr")
        if row_properties.find("w:tblHeader", NS) is None:
            etree.SubElement(row_properties, w_tag("tblHeader"))
    for row in rows:
        row_properties = get_or_prepend(row, "trPr")
        if row_properties.find("w:cantSplit", NS) is None:
            etree.SubElement(row_properties, w_tag("cantSplit"))
        cells = row.findall("w:tc", NS)
        for cell, width in zip(cells, widths, strict=True):
            set_cell_width(cell, width)
            for run in cell.findall(".//w:r", NS):
                set_run_size(run, font_size)


def transform_document(document_xml: bytes) -> bytes:
    parser = etree.XMLParser(remove_blank_text=False)
    root = etree.fromstring(document_xml, parser)
    set_page_layout(root)
    tables = root.findall(".//w:tbl", NS)
    if len(tables) != 2:
        raise ValueError("Expected the codebook and exclusion tables")
    set_table_layout(tables[0], [900, 3000, 2000, 2000, 6500], font_size=17)
    set_table_layout(tables[1], [8000, 2800, 2800], font_size=18)
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def rewrite_docx(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(source) as archive:
        document_xml = archive.read("word/document.xml")
        transformed = transform_document(document_xml)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as output:
            for item in archive.infolist():
                data = (
                    transformed
                    if item.filename == "word/document.xml"
                    else archive.read(item)
                )
                output.writestr(item, data)
    destination.write_bytes(buffer.getvalue())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rewrite_docx(args.source, args.destination)


if __name__ == "__main__":
    main()
