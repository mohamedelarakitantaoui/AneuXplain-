import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';

// ============================================
// Colors
// ============================================
const COLORS = {
  headerText: [26, 29, 39],       // #1A1D27
  bodyText: [51, 65, 85],         // #334155
  labelText: [100, 116, 139],     // #64748B
  accent: [74, 158, 255],         // #4A9EFF
  riskLow: [5, 150, 105],         // #059669
  riskModerate: [217, 119, 6],    // #D97706
  riskHigh: [220, 38, 38],        // #DC2626
  white: [255, 255, 255],
};

const RISK_COLORS = {
  LOW: COLORS.riskLow,
  MODERATE: COLORS.riskModerate,
  HIGH: COLORS.riskHigh,
  CRITICAL: COLORS.riskHigh,
  'N/A': COLORS.labelText,
};

const MARGIN = 20;
const PAGE_WIDTH = 210; // A4
const CONTENT_WIDTH = PAGE_WIDTH - MARGIN * 2;

function generateReportId() {
  const now = new Date();
  const ts = now.toISOString().replace(/[-:T]/g, '').slice(0, 14);
  const rand = Math.random().toString(36).substring(2, 8).toUpperCase();
  return `AX-${ts}-${rand}`;
}

function getRiskColor(level) {
  return RISK_COLORS[level] || COLORS.labelText;
}

// ============================================
// PDF Generator
// ============================================
export function generatePDFReport({
  filename,
  riskScore,
  riskLevel,
  morphologyData,
  clinicalReport,
  canvasImage,
  reportId,
  timestamp,
}) {
  const doc = new jsPDF({ unit: 'mm', format: 'a4' });
  const totalPagesExp = '{total_pages_count_string}';

  let y = MARGIN;

  // ---- PAGE 1: HEADER & RISK SUMMARY ----

  // Top banner title — two centered lines
  const centerX = PAGE_WIDTH / 2;
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(22);
  doc.setTextColor(...COLORS.headerText);
  doc.text('AneuXplain', centerX, y + 6, { align: 'center' });
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(14);
  doc.setTextColor(...COLORS.labelText);
  doc.text('Aneurysm Risk Analysis Report', centerX, y + 16, { align: 'center' });

  y += 22;

  // Blue separator line
  doc.setDrawColor(...COLORS.accent);
  doc.setLineWidth(0.5);
  doc.line(MARGIN, y, PAGE_WIDTH - MARGIN, y);
  y += 8;

  // Report metadata
  doc.setFontSize(9);
  doc.setTextColor(...COLORS.labelText);
  doc.text(`Patient Scan: ${filename}`, MARGIN, y);
  y += 5;
  doc.text(`Analysis Date: ${new Date(timestamp).toLocaleString()}`, MARGIN, y);
  y += 5;
  doc.text(`Report ID: ${reportId}`, MARGIN, y);
  y += 12;

  // ---- RUPTURE RISK ASSESSMENT ----
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(11);
  doc.setTextColor(...COLORS.headerText);
  doc.text('RUPTURE RISK ASSESSMENT', MARGIN, y);
  y += 8;

  // Risk score large number
  const percentage = Math.round((riskScore ?? 0) * 100);
  const riskColor = getRiskColor(riskLevel);

  doc.setFont('helvetica', 'bold');
  doc.setFontSize(48);
  doc.setTextColor(...riskColor);
  doc.text(`${percentage}%`, MARGIN, y + 14);

  // Risk level badge (next to score)
  const scoreWidth = doc.getTextWidth(`${percentage}%`);
  doc.setFontSize(14);
  doc.setTextColor(...riskColor);
  doc.text(riskLevel || 'UNKNOWN', MARGIN + scoreWidth + 6, y + 6);

  // Description line
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(9);
  doc.setTextColor(...COLORS.labelText);
  doc.text('Based on PointNet deep learning analysis of 3D surface geometry', MARGIN, y + 22);
  y += 30;

  // ---- 3D VIEWPORT SCREENSHOT ----
  if (canvasImage) {
    try {
      const imgProps = doc.getImageProperties(canvasImage);
      const imgWidth = CONTENT_WIDTH;
      const imgHeight = (imgProps.height / imgProps.width) * imgWidth;
      // Cap height to avoid overflow
      const maxHeight = 80;
      const finalHeight = Math.min(imgHeight, maxHeight);
      const finalWidth = imgHeight > maxHeight
        ? (imgProps.width / imgProps.height) * maxHeight
        : imgWidth;
      const imgX = MARGIN + (CONTENT_WIDTH - finalWidth) / 2;

      doc.addImage(canvasImage, 'PNG', imgX, y, finalWidth, finalHeight);
      y += finalHeight + 8;
    } catch {
      // Skip image if it fails
    }
  }

  // ---- MORPHOLOGICAL PARAMETERS TABLE ----
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(11);
  doc.setTextColor(...COLORS.headerText);
  doc.text('MORPHOLOGICAL PARAMETERS', MARGIN, y);
  y += 6;

  const parameters = clinicalReport?.parameters || [];

  const tableBody = parameters.map((param) => {
    const isNA = param.risk_level === 'N/A' || param.value == null;
    const value = isNA
      ? 'N/A'
      : typeof param.value === 'number'
        ? (param.value < 1 ? param.value.toFixed(3) : param.value < 100 ? param.value.toFixed(2) : param.value.toFixed(1))
        : String(param.value);
    const unit = isNA ? '' : (param.unit || '');
    const risk = param.risk_level || 'N/A';
    const normalRange = isNA ? 'Requires parent vessel' : (param.normal_range || '-');

    return [param.parameter, value, unit, risk, normalRange];
  });

  // Footer occupies ~25mm from bottom; keep 15mm gap above it
  const FOOTER_ZONE = 40; // reserve bottom 40mm (25mm footer + 15mm gap)
  const PAGE_HEIGHT = 297; // A4 height

  autoTable(doc, {
    startY: y,
    head: [['Parameter', 'Value', 'Unit', 'Risk Level', 'Normal Range']],
    body: tableBody,
    margin: { left: MARGIN, right: MARGIN, bottom: FOOTER_ZONE },
    styles: {
      font: 'helvetica',
      fontSize: 9,
      cellPadding: 3,
      textColor: COLORS.bodyText,
      lineColor: [230, 230, 230],
      lineWidth: 0.2,
    },
    headStyles: {
      fillColor: [240, 243, 248],
      textColor: COLORS.headerText,
      fontStyle: 'bold',
      fontSize: 8,
    },
    columnStyles: {
      0: { fontStyle: 'bold', cellWidth: 42 },
      1: { halign: 'center', cellWidth: 25 },
      2: { halign: 'center', cellWidth: 18 },
      3: { halign: 'center', cellWidth: 28 },
      4: { cellWidth: 'auto' },
    },
    didParseCell: (data) => {
      // Color the Risk Level column
      if (data.section === 'body' && data.column.index === 3) {
        const level = data.cell.raw;
        const color = getRiskColor(level);
        data.cell.styles.textColor = color;
        data.cell.styles.fontStyle = 'bold';
      }
      // Gray out N/A values
      if (data.section === 'body' && data.column.index === 1 && data.cell.raw === 'N/A') {
        data.cell.styles.textColor = COLORS.labelText;
        data.cell.styles.fontStyle = 'italic';
      }
    },
  });

  // Clinical Findings always starts on a new page
  doc.addPage();
  y = MARGIN;

  doc.setFont('helvetica', 'bold');
  doc.setFontSize(11);
  doc.setTextColor(...COLORS.headerText);
  doc.text('CLINICAL FINDINGS', MARGIN, y);
  y += 8;

  // Filter parameters with MODERATE or HIGH risk
  const flaggedParams = parameters.filter(
    (p) => p.risk_level === 'MODERATE' || p.risk_level === 'HIGH'
  );

  if (flaggedParams.length === 0) {
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(10);
    doc.setTextColor(...COLORS.bodyText);
    doc.text('No parameters flagged as moderate or high risk.', MARGIN, y);
    y += 12;
  } else {
    for (const param of flaggedParams) {
      // Check if we need a new page (stay above footer zone)
      if (y > PAGE_HEIGHT - FOOTER_ZONE) {
        doc.addPage();
        y = MARGIN;
      }

      const color = getRiskColor(param.risk_level);

      // Parameter name as sub-header
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(10);
      doc.setTextColor(...color);
      doc.text(`${param.parameter}  [${param.risk_level}]`, MARGIN, y);
      y += 6;

      // Explanation
      if (param.explanation) {
        doc.setFont('helvetica', 'normal');
        doc.setFontSize(9);
        doc.setTextColor(...COLORS.bodyText);
        const lines = doc.splitTextToSize(param.explanation, CONTENT_WIDTH - 5);
        doc.text(lines, MARGIN + 3, y);
        y += lines.length * 4.2 + 3;
      }

      // Clinical significance
      if (param.clinical_significance) {
        doc.setFont('helvetica', 'italic');
        doc.setFontSize(8.5);
        doc.setTextColor(...COLORS.labelText);
        const sigLines = doc.splitTextToSize(param.clinical_significance, CONTENT_WIDTH - 5);
        doc.text(sigLines, MARGIN + 3, y);
        y += sigLines.length * 3.8 + 6;
      }

      // Light separator between findings
      doc.setDrawColor(230, 230, 230);
      doc.setLineWidth(0.15);
      doc.line(MARGIN, y, PAGE_WIDTH - MARGIN, y);
      y += 6;
    }
  }

  // ---- SUMMARY ----
  if (clinicalReport?.summary) {
    if (y > PAGE_HEIGHT - FOOTER_ZONE) {
      doc.addPage();
      y = MARGIN;
    }

    doc.setFont('helvetica', 'bold');
    doc.setFontSize(11);
    doc.setTextColor(...COLORS.headerText);
    doc.text('SUMMARY', MARGIN, y);
    y += 7;

    doc.setFont('helvetica', 'normal');
    doc.setFontSize(9.5);
    doc.setTextColor(...COLORS.bodyText);
    const summaryLines = doc.splitTextToSize(clinicalReport.summary, CONTENT_WIDTH);
    doc.text(summaryLines, MARGIN, y);
  }

  // ---- FOOTER on all pages ----
  const pageCount = doc.internal.getNumberOfPages();
  for (let i = 1; i <= pageCount; i++) {
    doc.setPage(i);
    const pageH = doc.internal.pageSize.getHeight();

    // Separator line
    doc.setDrawColor(...COLORS.accent);
    doc.setLineWidth(0.3);
    doc.line(MARGIN, pageH - 20, PAGE_WIDTH - MARGIN, pageH - 20);

    // Footer text
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(7);
    doc.setTextColor(...COLORS.labelText);
    doc.text(
      'Generated by AneuXplain  |  Al Akhawayn University  |  For research purposes only',
      MARGIN,
      pageH - 15
    );
    doc.text(
      'This report is generated by an AI system and should not replace clinical judgment.',
      MARGIN,
      pageH - 11
    );

    // Page number
    doc.setFontSize(7);
    doc.text(
      `Page ${i} of ${pageCount}`,
      PAGE_WIDTH - MARGIN,
      pageH - 11,
      { align: 'right' }
    );
  }

  return doc;
}

// ============================================
// Export helpers
// ============================================

export function buildExportData({
  filename,
  riskScore,
  riskLevel,
  morphologyData,
  clinicalReport,
  reportId,
  timestamp,
}) {
  return {
    report_id: reportId,
    timestamp,
    filename,
    risk_score: riskScore,
    risk_level: riskLevel,
    morphology: morphologyData,
    clinical_report: clinicalReport,
  };
}

export function downloadBlob(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export function exportAll({
  filename,
  riskScore,
  riskLevel,
  morphologyData,
  clinicalReport,
  canvasImage,
}) {
  const reportId = generateReportId();
  const timestamp = new Date().toISOString();
  const dateStr = timestamp.split('T')[0];
  const cleanName = (filename || 'scan').replace('.obj', '');

  // 1. Generate PDF
  const doc = generatePDFReport({
    filename,
    riskScore,
    riskLevel,
    morphologyData,
    clinicalReport,
    canvasImage,
    reportId,
    timestamp,
  });
  const pdfBlob = doc.output('blob');
  downloadBlob(pdfBlob, `AneuXplain_Report_${cleanName}_${dateStr}.pdf`);

  // 2. Generate JSON
  const jsonData = buildExportData({
    filename,
    riskScore,
    riskLevel,
    morphologyData,
    clinicalReport,
    reportId,
    timestamp,
  });
  const jsonBlob = new Blob([JSON.stringify(jsonData, null, 2)], { type: 'application/json' });
  downloadBlob(jsonBlob, `AneuXplain_Data_${cleanName}_${dateStr}.json`);
}
