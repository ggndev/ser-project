// Markdown (the subset used in docs/) -> Word .docx, one combined document per folder.
// Re-run after editing the Markdown:  cd tools && npm install docx && node md_to_docx.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, WidthType, ShadingType,
  BorderStyle, AlignmentType, LevelFormat, PageNumber, Footer, Header, TabStopType,
} = require("docx");

const ROOT = "/home/yashu/Desktop/Speech Project/docs";
const TEXT_W = 9638;                       // A4 (11906) minus 2 x 1134 (2 cm) margins, in DXA
const BODY_FONT = "Calibri";
const CODE_FONT = "Consolas";

const JOBS = [
  { out: "01-ml/01-ml-documentation.docx", title: "EmoSense — Machine-Learning Documentation",
    subtitle: "Notebook ser_report_notebook.ipynb explained: data, preprocessing, models, evaluation, export",
    files: ["01-ml/README.md", "01-ml/01-data-and-labels.md", "01-ml/02-preprocessing-augmentation.md", "01-ml/03-models.md", "01-ml/04-training-evaluation.md", "01-ml/05-export-and-realtime.md"] },
  { out: "02-backend/02-backend-documentation.docx", title: "EmoSense — Backend Documentation",
    subtitle: "FastAPI + ONNX Runtime service: app.py, requirements and Dockerfile explained",
    files: ["02-backend/README.md"] },
  { out: "03-deployment/03-deployment-documentation.docx", title: "EmoSense — Deployment Documentation",
    subtitle: "From the model file to a working phone app: GitHub, Render, UptimeRobot and API tests",
    files: ["03-deployment/README.md"] },
  { out: "04-android/04-android-documentation.docx", title: "EmoSense — Android App Documentation",
    subtitle: "Kotlin + Jetpack Compose app: recorder, on-device login, API client, wellness messages",
    files: ["04-android/README.md"] },
];

// ------------------------------------------------------------------ inline markdown -> TextRuns
function inline(text, base = {}) {
  const runs = [];
  const push = (t, extra = {}) => { if (t) runs.push(new TextRun({ text: t, font: BODY_FONT, ...base, ...extra })); };
  let rest = text;
  const patterns = [
    { re: /`([^`]+)`/, make: (m) => runs.push(new TextRun({ text: m[1], font: CODE_FONT, size: (base.size || 21) - 1, shading: { type: ShadingType.CLEAR, fill: "EEEEEE", color: "auto" }, bold: base.bold, italics: base.italics, color: base.color })) },
    { re: /\*\*(.+?)\*\*/, make: (m) => runs.push(...inline(m[1], { ...base, bold: true })) },
    { re: /\*([^*\s][^*]*?)\*/, make: (m) => runs.push(...inline(m[1], { ...base, italics: true })) },
    { re: /\[([^\]]+)\]\(([^)]+)\)/, make: (m) => runs.push(...inline(m[1], { ...base })) },
  ];
  while (rest.length) {
    let best = null;
    for (const p of patterns) {
      const m = p.re.exec(rest);
      if (m && (best === null || m.index < best.m.index)) best = { p, m };
    }
    if (!best) { push(rest); break; }
    push(rest.slice(0, best.m.index));
    best.p.make(best.m);
    rest = rest.slice(best.m.index + best.m[0].length);
  }
  return runs;
}

// ------------------------------------------------------------------ block parsing
function splitRow(line) {
  const cells = []; let cur = ""; let inCode = false;
  const s = line.trim().replace(/^\|/, "").replace(/\|$/, "");
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if (ch === "`") inCode = !inCode;
    if (ch === "|" && !inCode && s[i - 1] !== "\\") { cells.push(cur.trim()); cur = ""; } else cur += ch;
  }
  cells.push(cur.trim());
  return cells.map((c) => c.replace(/\\\|/g, "|"));
}

let numberedCount = 0;
const numberedRefs = [];

function codeParagraphs(lines, lang) {
  const maxLen = Math.max(...lines.map((l) => l.length), 1);
  let size = 17;                                   // 8.5 pt
  if (!lang) size = maxLen > 100 ? 14 : maxLen > 88 ? 15 : 16;   // diagrams: shrink to fit the page width
  else if (maxLen > 110) size = 15;
  return lines.map((l, i) => new Paragraph({
    shading: { type: ShadingType.CLEAR, fill: "F4F4F4", color: "auto" },
    border: { left: { style: BorderStyle.SINGLE, size: 12, color: "9AA7B5", space: 6 } },
    indent: { left: 120 },
    spacing: { before: i === 0 ? 80 : 0, after: i === lines.length - 1 ? 160 : 0, line: 236 },
    children: [new TextRun({ text: l.length ? l : " ", font: CODE_FONT, size })],
  }));
}

function tableBlock(rows) {
  const header = rows[0]; const body = rows.slice(2);
  const all = [header, ...body];
  const n = header.length;
  const weights = [];
  for (let c = 0; c < n; c++) {
    const m = Math.max(...all.map((r) => (r[c] || "").replace(/[*`]/g, "").length));
    weights.push(Math.min(Math.max(m, 7), 48));
  }
  const tot = weights.reduce((a, b) => a + b, 0);
  const widths = weights.map((w) => Math.floor((w / tot) * TEXT_W));
  widths[n - 1] += TEXT_W - widths.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: "B7C0CA" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const mkRow = (cells, isHeader) => new TableRow({
    tableHeader: isHeader, cantSplit: true,
    children: Array.from({ length: n }, (_, c) => new TableCell({
      width: { size: widths[c], type: WidthType.DXA }, borders,
      shading: isHeader ? { type: ShadingType.CLEAR, fill: "DCE6F2", color: "auto" } : undefined,
      margins: { top: 50, bottom: 50, left: 90, right: 90 },
      children: [new Paragraph({ spacing: { after: 0 }, children: inline(cells[c] || "", { size: 18, bold: isHeader }) })],
    })),
  });
  return [new Table({ width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: widths, rows: [mkRow(header, true), ...body.map((r) => mkRow(r, false))] }),
          new Paragraph({ spacing: { after: 120 }, children: [] })];
}

const HEADINGS = [HeadingLevel.HEADING_1, HeadingLevel.HEADING_2, HeadingLevel.HEADING_3, HeadingLevel.HEADING_4, HeadingLevel.HEADING_4, HeadingLevel.HEADING_4];

function parseMarkdown(md, isFirstFile) {
  const lines = md.replace(/\r/g, "").split("\n");
  const out = []; let para = []; let i = 0; let firstH1 = true;
  const flush = () => { if (para.length) { out.push(new Paragraph({ spacing: { after: 120, line: 276 }, children: inline(para.join(" ").trim(), { size: 21 }) })); para = []; } };
  while (i < lines.length) {
    const line = lines[i];
    let m;
    if ((m = /^```(\w*)\s*$/.exec(line))) {                       // fenced code
      flush(); const lang = m[1]; const buf = []; i++;
      while (i < lines.length && !/^```\s*$/.test(lines[i])) { buf.push(lines[i]); i++; }
      i++; out.push(...codeParagraphs(buf, lang)); continue;
    }
    if ((m = /^(#{1,6})\s+(.*)$/.exec(line))) {                   // heading
      flush(); const lvl = m[1].length;
      out.push(new Paragraph({ heading: HEADINGS[lvl - 1], pageBreakBefore: lvl === 1 && !(firstH1 && isFirstFile ? false : false) && true, children: inline(m[2], {}) }));
      if (lvl === 1) firstH1 = false;
      i++; continue;
    }
    if (/^\|.*\|\s*$/.test(line) && i + 1 < lines.length && /^\|[\s:|-]+\|\s*$/.test(lines[i + 1])) {   // table
      flush(); const rows = [];
      while (i < lines.length && /^\|.*\|\s*$/.test(lines[i])) { rows.push(splitRow(lines[i])); i++; }
      out.push(...tableBlock(rows)); continue;
    }
    if (/^---+\s*$/.test(line)) { flush(); out.push(new Paragraph({ border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "B7C0CA", space: 1 } }, spacing: { after: 120 }, children: [] })); i++; continue; }
    if (/^>\s?/.test(line)) {                                      // block quote
      flush(); const buf = [];
      while (i < lines.length && /^>\s?/.test(lines[i])) { buf.push(lines[i].replace(/^>\s?/, "")); i++; }
      out.push(new Paragraph({ indent: { left: 360 }, border: { left: { style: BorderStyle.SINGLE, size: 18, color: "7C5CFF", space: 8 } }, spacing: { after: 140 },
        children: inline(buf.join(" ").trim(), { size: 21, italics: true }) })); continue;
    }
    if ((m = /^(\s*)([*-]|\d+\.)\s+(.*)$/.exec(line))) {          // list (consecutive items)
      flush(); const numbered = /\d+\./.test(m[2]);
      const ref = numbered ? "num" + (++numberedCount) : "bullets"; if (numbered) numberedRefs.push(ref);
      while (i < lines.length) {
        const mm = /^(\s*)([*-]|\d+\.)\s+(.*)$/.exec(lines[i]);
        if (!mm || /\d+\./.test(mm[2]) !== numbered) break;
        let text = mm[3]; const level = Math.min(Math.floor(mm[1].length / 2), 2); i++;
        while (i < lines.length && /^\s{2,}\S/.test(lines[i]) && !/^(\s*)([*-]|\d+\.)\s+/.test(lines[i])) { text += " " + lines[i].trim(); i++; }
        out.push(new Paragraph({ numbering: { reference: ref, level }, spacing: { after: 60, line: 276 }, children: inline(text, { size: 21 }) }));
      }
      out.push(new Paragraph({ spacing: { after: 60 }, children: [] })); continue;
    }
    if (/^\s*$/.test(line)) { flush(); i++; continue; }
    para.push(line.trim()); i++;
  }
  flush();
  return out;
}

// ------------------------------------------------------------------ build each document
function build(job) {
  numberedCount = 0; numberedRefs.length = 0;
  const body = [];
  // cover block
  body.push(new Paragraph({ spacing: { before: 2400, after: 200 }, children: [new TextRun({ text: job.title, font: BODY_FONT, size: 52, bold: true, color: "2B2F6B" })] }));
  body.push(new Paragraph({ spacing: { after: 400 }, border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: "7C5CFF", space: 6 } }, children: [new TextRun({ text: job.subtitle, font: BODY_FONT, size: 26, color: "555555" })] }));
  body.push(new Paragraph({ spacing: { after: 120 }, children: [new TextRun({ text: "Contents of this document", font: BODY_FONT, size: 24, bold: true })] }));
  job.files.forEach((f) => body.push(new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 40 }, children: [new TextRun({ text: f, font: CODE_FONT, size: 20 })] })));
  body.push(new Paragraph({ spacing: { before: 400 }, children: [new TextRun({ text: "Generated from the Markdown files in the docs folder (the Markdown files remain the source of truth).", font: BODY_FONT, size: 18, italics: true, color: "777777" })] }));
  job.files.forEach((f, idx) => body.push(...parseMarkdown(fs.readFileSync(path.join(ROOT, f), "utf8"), idx === 0)));

  const numbering = { config: [
    { reference: "bullets", levels: [0, 1, 2].map((lv) => ({ level: lv, format: LevelFormat.BULLET, text: ["•", "–", "·"][lv], alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 540 + lv * 360, hanging: 270 } } } })) },
    ...numberedRefs.map((ref) => ({ reference: ref, levels: [0, 1, 2].map((lv) => ({ level: lv, format: LevelFormat.DECIMAL, text: `%${lv + 1}.`, alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 540 + lv * 360, hanging: 360 } } } })) })),
  ] };

  const heading = (size, color, before, after) => ({ run: { font: BODY_FONT, size, bold: true, color }, paragraph: { spacing: { before, after } } });
  return new Document({
    creator: "EmoSense project", title: job.title,
    styles: {
      default: { document: { run: { font: BODY_FONT, size: 21 } } },
      paragraphStyles: [
        { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, ...heading(36, "2B2F6B", 240, 160), },
        { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, ...heading(28, "3B4A8F", 280, 120) },
        { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, ...heading(24, "3B4A8F", 220, 100) },
        { id: "Heading4", name: "Heading 4", basedOn: "Normal", next: "Normal", quickFormat: true, ...heading(22, "444444", 180, 80) },
      ].map((s, i) => ({ ...s, paragraph: { ...s.paragraph, outlineLevel: i, keepNext: true } })),
    },
    numbering,
    sections: [{
      properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, right: 1134, bottom: 1134, left: 1134 } } },
      headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: job.title, font: BODY_FONT, size: 16, color: "888888" })] })] }) },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: ["Page ", PageNumber.CURRENT, " of ", PageNumber.TOTAL_PAGES], font: BODY_FONT, size: 18, color: "888888" })] })] }) },
      children: body,
    }],
  });
}

(async () => {
  for (const job of JOBS) {
    const doc = build(job);
    const buf = await Packer.toBuffer(doc);
    const outPath = path.join(ROOT, job.out);
    fs.writeFileSync(outPath, buf);
    console.log("wrote", outPath, (buf.length / 1024).toFixed(0) + " KB");
  }
})();
