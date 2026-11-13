"""Build report.pdf from report.md using standard-library Python and LibreOffice.

This renderer supports the small Markdown subset used by this report. It never
imports the experiment package or evaluates the battery model. LibreOffice exports the intermediate ODT to PDF. No Word output is produced.
"""
from pathlib import Path
from html import escape
import argparse
import shutil
import subprocess
import tempfile
import hashlib
import json
import re
import struct
import zipfile

HERE = Path(__file__).resolve().parents[1] / 'report'
BUILD = HERE.parent / 'artifacts/report-build'
NS = ('xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
      'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
      'xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0" '
      'xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0" '
      'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" '
      'xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0" '
      'xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0" '
      'xmlns:xlink="http://www.w3.org/1999/xlink"')


def inline(value):
    parts = re.split(r'(\[[^\]]+\]\(https?://[^)]+\)|\*\*[^*]+\*\*)', value)
    output = []
    for part in parts:
        link = re.fullmatch(r'\[([^\]]+)\]\((https?://[^)]+)\)', part)
        if link:
            output.append(f'<text:a xlink:type="simple" xlink:href="{escape(link[2], quote=True)}">{escape(link[1])}</text:a>')
        elif part.startswith('**') and part.endswith('**'):
            output.append(f'<text:span text:style-name="Strong">{escape(part[2:-2])}</text:span>')
        else:
            output.append(escape(part))
    return ''.join(output)


def paragraph(value, style='Body'):
    return f'<text:p text:style-name="{style}">{inline(value)}</text:p>'


def style(name, parent='Body', para='', text=''):
    return (f'<style:style style:name="{name}" style:family="paragraph" style:parent-style-name="{parent}">'
            f'<style:paragraph-properties {para}/><style:text-properties {text}/></style:style>')


def build():
    lines = (HERE / 'report.md').read_text().splitlines()
    body, extra_styles, images = [], [], {}
    page_break = False
    references = False
    table_number = 0
    blocks = 1
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line == '<!-- page -->':
            page_break = True
            blocks += 1
            i += 1
            continue
        if line.startswith('#'):
            level = len(line) - len(line.lstrip('#'))
            value = line[level:].strip()
            name = 'Title' if level == 1 else 'H1' if level == 2 else 'H2'
            if page_break:
                name += 'Break'
                page_break = False
            references = value == 'References'
            body.append(paragraph(value, name))
            i += 1
            continue
        image = re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)', line)
        if image:
            path = HERE / image[2]
            data = path.read_bytes()
            w, h = struct.unpack('>II', data[16:24])
            width = 17.6
            height = width * h / w
            images[image[2]] = data
            body.append('<text:p text:style-name="Figure">'
                        f'<draw:frame draw:style-name="Image" draw:name="{escape(path.stem)}" '
                        f'text:anchor-type="as-char" svg:width="{width:.3f}cm" svg:height="{height:.3f}cm">'
                        f'<draw:image xlink:href="{escape(image[2])}" xlink:type="simple" '
                        'xlink:show="embed" xlink:actuate="onLoad"/>'
                        f'<svg:desc>{escape(image[1])}</svg:desc></draw:frame></text:p>')
            i += 1
            continue
        if line.startswith('|'):
            table_number += 1
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                cells = [x.strip() for x in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?', c) for c in cells):
                    rows.append(cells)
                i += 1
            weights = {1:[1.8,5.2], 2:[1.5,2.6,2.1,.8],
                       3:[1.0,1.0,1.0,2.2,2.2], 4:[1.0,.9,2.5,2.5],
                       5:[1.6,1.5,1.6,1.1,1.0]}[table_number]
            assert all(len(row) == len(weights) for row in rows)
            cols = []
            for k, weight in enumerate(weights):
                name = f'T{table_number}C{k}'
                extra_styles.append(f'<style:style style:name="{name}" style:family="table-column">'
                                    f'<style:table-column-properties style:column-width="{17.6*weight/sum(weights):.3f}cm"/>'
                                    '</style:style>')
                cols.append(f'<table:table-column table:style-name="{name}"/>')
            table = [f'<table:table table:name="Table{table_number}" table:style-name="Table">', *cols]
            for j, row in enumerate(rows):
                if j == 0:
                    table.append('<table:table-header-rows>')
                table.append('<table:table-row table:style-name="Row">')
                for value in row:
                    table.append(f'<table:table-cell table:style-name="{"HeaderCell" if j==0 else "Cell"}" office:value-type="string">'
                                 + paragraph(value, 'TableHead' if j==0 else 'TableBody') + '</table:table-cell>')
                table.append('</table:table-row>')
                if j == 0:
                    table.append('</table:table-header-rows>')
            table.append('</table:table>')
            body.extend(table)
            continue
        values = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(('#', '|', '![', '<!--')):
            values.append(lines[i].strip())
            i += 1
        value = ' '.join(values)
        if value.startswith('> '):
            value, name = value[2:], 'Equation'
        elif re.match(r'Figure \d+\.', value):
            name = 'Caption'
        elif re.match(r'Table \d+\.', value):
            name = 'TableCaption'
        elif references:
            name = 'Reference'
        elif len(body) < 4:
            name = 'Meta'
        else:
            name = 'Body'
        if page_break:
            name = 'BodyBreak'
            page_break = False
        body.append(paragraph(value, name))


    styles = [
        style('Body', 'Standard', 'fo:margin-bottom="0.16cm" fo:line-height="112%" fo:orphans="2" fo:widows="2" fo:text-align="justify"',
              'fo:font-family="Liberation Serif" fo:font-size="11pt" fo:color="#161616"'),
        style('BodyBreak', para='fo:break-before="page"'),
        style('Title', para='fo:margin-bottom="0.24cm" fo:line-height="108%" fo:keep-with-next="always" fo:text-align="left"',
              text='fo:font-family="Liberation Sans" fo:font-size="19pt" fo:font-weight="bold" fo:color="#17334B"'),
        style('Meta', para='fo:margin-bottom="0.13cm" fo:text-align="left"', text='fo:font-size="9.5pt"'),
        style('H1', para='fo:margin-top="0.27cm" fo:margin-bottom="0.16cm" fo:keep-with-next="always" fo:text-align="left"',
              text='fo:font-family="Liberation Sans" fo:font-size="13pt" fo:font-weight="bold" fo:color="#17334B"'),
        style('H1Break', 'H1', para='fo:break-before="page" fo:margin-top="0cm"'),
        style('H2', para='fo:margin-top="0.21cm" fo:margin-bottom="0.12cm" fo:keep-with-next="always" fo:text-align="left"',
              text='fo:font-size="11.5pt" fo:font-weight="bold"'),
        style('H2Break', 'H2', para='fo:break-before="page" fo:margin-top="0cm"'),
        style('Equation', para='fo:text-align="center" fo:margin-top="0.08cm" fo:margin-bottom="0.18cm"', text='fo:font-size="10.5pt"'),
        style('Figure', para='fo:text-align="center" fo:keep-with-next="always" fo:margin-top="0.14cm" fo:margin-bottom="0.08cm"'),
        style('Caption', para='fo:margin-bottom="0.20cm" fo:line-height="105%" fo:text-align="left"', text='fo:font-size="9pt"'),
        style('TableCaption', 'Caption', para='fo:keep-with-next="always" fo:margin-top="0.12cm"'),
        style('TableBody', para='fo:margin-bottom="0cm" fo:line-height="105%" fo:text-align="left"', text='fo:font-size="9pt"'),
        style('TableHead', 'TableBody', text='fo:font-weight="bold"'),
        style('Reference', para='fo:margin-bottom="0.13cm" fo:line-height="105%" fo:text-align="left"', text='fo:font-size="9.5pt"'),
        style('Footer', para='fo:text-align="center"', text='fo:font-family="Liberation Sans" fo:font-size="8.5pt" fo:color="#626262"'),
        '<style:style style:name="Strong" style:family="text"><style:text-properties fo:font-weight="bold"/></style:style>',
        '<style:style style:name="Table" style:family="table"><style:table-properties style:width="17.6cm" table:align="center" fo:margin-bottom="0.16cm"/></style:style>',
        '<style:style style:name="Cell" style:family="table-cell"><style:table-cell-properties fo:padding="0.11cm" fo:border-bottom="0.4pt solid #BBC5CE"/></style:style>',
        '<style:style style:name="HeaderCell" style:family="table-cell"><style:table-cell-properties fo:padding="0.11cm" fo:background-color="#E8EEF3" fo:border-bottom="0.8pt solid #49647B"/></style:style>',
        '<style:style style:name="Row" style:family="table-row"><style:table-row-properties fo:keep-together="always"/></style:style>',
        '<style:style style:name="Image" style:family="graphic"><style:graphic-properties style:vertical-pos="top" style:vertical-rel="baseline"/></style:style>',
    ]
    content = (f'<?xml version="1.0" encoding="UTF-8"?><office:document-content {NS} office:version="1.2">'
               '<office:automatic-styles>'+''.join(styles+extra_styles)+'</office:automatic-styles>'
               '<office:body><office:text>'+''.join(body)+'</office:text></office:body></office:document-content>')
    page = ('<style:page-layout style:name="A4"><style:page-layout-properties fo:page-width="21cm" fo:page-height="29.7cm" '
            'style:print-orientation="portrait" fo:margin-top="1.65cm" fo:margin-bottom="1.45cm" '
            'fo:margin-left="1.7cm" fo:margin-right="1.7cm"/>'
            '<style:footer-style><style:header-footer-properties fo:min-height="0.4cm" fo:margin-top="0.20cm"/></style:footer-style>'
            '</style:page-layout>')
    styles_xml = (f'<?xml version="1.0" encoding="UTF-8"?><office:document-styles {NS} office:version="1.2">'
                  '<office:styles>'+''.join(styles)+'</office:styles><office:automatic-styles>'+page+'</office:automatic-styles>'
                  '<office:master-styles><style:master-page style:name="Standard" style:page-layout-name="A4"><style:footer>'
                  '<text:p text:style-name="Footer">DSA4212 · Geometry and numerical search   |   '
                  '<text:page-number text:select-page="current">1</text:page-number></text:p>'
                  '</style:footer></style:master-page></office:master-styles></office:document-styles>')
    files = {'content.xml':content.encode(), 'styles.xml':styles_xml.encode(), **images}
    manifest = ['<?xml version="1.0" encoding="UTF-8"?><manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0" manifest:version="1.2">',
                '<manifest:file-entry manifest:full-path="/" manifest:media-type="application/vnd.oasis.opendocument.text"/>']
    for name in files:
        mime = 'image/png' if name.endswith('.png') else 'text/xml'
        manifest.append(f'<manifest:file-entry manifest:full-path="{name}" manifest:media-type="{mime}"/>')
    manifest.append('</manifest:manifest>')
    with zipfile.ZipFile(BUILD/'report.odt','w',zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('mimetype','application/vnd.oasis.opendocument.text',compress_type=zipfile.ZIP_STORED)
        archive.writestr('META-INF/manifest.xml',''.join(manifest))
        for name,data in files.items():
            archive.writestr(name,data)
    summary = {'source':'report.md','explicit_page_sections':blocks,'tables':table_number,'figures':list(images),
               'source_sha256':hashlib.sha256((HERE/'report.md').read_bytes()).hexdigest()}
    (BUILD/'source_manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--soffice', default='soffice', help='LibreOffice executable (or absolute path)')
    args = parser.parse_args()
    BUILD.mkdir(parents=True, exist_ok=True)
    build()
    with tempfile.TemporaryDirectory(prefix='bo-es-pdf-') as directory:
        temp = Path(directory)
        subprocess.run([args.soffice, '-env:UserInstallation=' + (temp / 'profile').as_uri(),
                        '--headless', '--convert-to', 'pdf:writer_pdf_Export', '--outdir', str(temp),
                        str(BUILD / 'report.odt')], check=True, timeout=120)
        pdf = temp / 'report.pdf'
        if not pdf.is_file() or pdf.stat().st_size == 0:
            raise RuntimeError('LibreOffice did not produce a PDF')
        shutil.copyfile(pdf, HERE / 'report.pdf')
    print('Exported report/report.pdf; intermediates are in artifacts/report-build/.')
