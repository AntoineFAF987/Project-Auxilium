from pathlib import Path
from zipfile import ZipFile

from rag_core.document_parsing import parse_document
from rag_core.utils import read_supported_text


def _workbook(path: Path):
    shared = """<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
    <si><t>DN</t></si><si><t>KG2 2022</t></si><si><t>KG2 2023</t></si><si><t>50</t></si>
    </sst>"""
    workbook = """<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheets><sheet name="Prix d'achat HT" sheetId="1"/></sheets></workbook>"""
    sheet = """<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>
    <row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c><c r="C1" t="s"><v>2</v></c></row>
    <row r="2"><c r="A2" t="s"><v>3</v></c><c r="B2"><v>47.95</v></c><c r="C2"><f>B2*1.13</f><v>54.21</v></c></row>
    </sheetData></worksheet>"""
    with ZipFile(path, "w") as archive:
        archive.writestr("xl/sharedStrings.xml", shared)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/worksheets/sheet1.xml", sheet)


def test_xlsx_rows_are_rendered_with_headers_and_value_origin(tmp_path):
    path = tmp_path / "tarif.xlsx"
    _workbook(path)
    text = read_supported_text(str(path))
    assert "Prix d'achat HT | DN 50" in text
    assert "DN 50" in text
    assert "KG2 2022: 47.95" in text
    assert "KG2 2023: 54.21 [computed]" in text

    document = parse_document(str(path))
    row = next(block for block in document.blocks if block.block_type == "table_row")
    assert row.source_metadata["sheet_name"] == "Prix d'achat HT"
    assert row.source_metadata["row_key"] == "50"
    computed = next(cell for cell in row.source_metadata["structured_cells"] if cell["column_key"] == "KG2 2023")
    assert computed["cell_value"] == "54.21"
    assert computed["value_origin"] == "computed"
