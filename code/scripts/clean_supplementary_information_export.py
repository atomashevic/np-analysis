"""Remove the Google Docs tab-title cover from an exported SI DOCX."""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

from lxml import etree


WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": WORD_NS}


def paragraph_text(paragraph: etree._Element) -> str:
    return "".join(paragraph.xpath(".//w:t/text()", namespaces=NS))


def remove_tab_title_cover(document_xml: bytes, expected_title: str) -> bytes:
    parser = etree.XMLParser(remove_blank_text=False)
    root = etree.fromstring(document_xml, parser)
    body = root.find("w:body", NS)
    if body is None:
        raise ValueError("DOCX has no document body")

    paragraphs = body.findall("w:p", NS)
    if len(paragraphs) < 2:
        raise ValueError("DOCX has too few paragraphs to contain a tab-title cover")
    if paragraph_text(paragraphs[0]) != expected_title:
        raise ValueError("First paragraph does not match the expected tab title")
    if paragraph_text(paragraphs[1]) != expected_title:
        raise ValueError("Second paragraph does not match the document title")

    body.remove(paragraphs[0])
    return etree.tostring(
        root,
        xml_declaration=True,
        encoding="UTF-8",
        standalone=True,
    )


def rewrite_docx(source: Path, destination: Path, expected_title: str) -> None:
    with zipfile.ZipFile(source) as archive:
        transformed = remove_tab_title_cover(
            archive.read("word/document.xml"),
            expected_title,
        )
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
    parser.add_argument(
        "--title",
        default="Supplementary Information",
        help="Expected Google Docs tab and document title.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rewrite_docx(args.source, args.destination, args.title)


if __name__ == "__main__":
    main()
