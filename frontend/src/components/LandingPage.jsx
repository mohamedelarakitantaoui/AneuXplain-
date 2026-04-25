import { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { Shield, Upload, ArrowRight } from 'lucide-react';
import '../assets/css/bootstrap.min.css';
import '../assets/css/animate.css';
import '../assets/css/custom-animation.css';
import '../assets/css/font-awesome-pro.css';
import '../assets/css/spacing.css';
import '../assets/css/style.css';


const inView = (delay = 0) => ({
  initial: { opacity: 0, y: 32 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true },
  transition: { duration: 0.65, ease: [0.22, 1, 0.36, 1], delay },
});

export default function LandingPage({ onOpenApp, onStartMesh, onStartDicom }) {
  const scrollRef = useRef(null);
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const onScroll = () => setScrolled(el.scrollTop > 60);
    el.addEventListener('scroll', onScroll, { passive: true });
    return () => el.removeEventListener('scroll', onScroll);
  }, []);

  return (
    <div ref={scrollRef} style={{ position: 'fixed', inset: 0, zIndex: 200, overflowY: 'auto', background: '#fff', '--tp-theme-1': '#dc2626', '--tp-common-blue': '#dc2626' }}>

      {/* ── HEADER ───────────────────────────────────────── */}
      <div
        style={{
          position: 'fixed', top: 0, left: 0, right: 0, zIndex: 300,
          background: scrolled ? 'rgba(255,255,255,0.95)' : 'transparent',
          backdropFilter: scrolled ? 'blur(16px)' : 'none',
          WebkitBackdropFilter: scrolled ? 'blur(16px)' : 'none',
          borderBottom: scrolled ? '1px solid #EBECF0' : 'none',
          transition: 'background 0.3s, border-color 0.3s, backdrop-filter 0.3s',
        }}
      >
        <div className="container">
          <div className="row align-items-center" style={{ height: 72 }}>

            {/* Logo */}
            <div className="col-xxl-3 col-xl-3 col-lg-3 col-md-4 col-6">
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <img src="/logoAneuX.png" alt="AneuXplain"
                  style={{ width: 36, height: 36, borderRadius: 10, objectFit: 'contain' }} />
                <span style={{
                  fontSize: 17, fontWeight: 700,
                  color: '#000229',
                  fontFamily: 'var(--tp-ff-urban)', letterSpacing: '-0.01em',
                }}>
                  AneuXplain
                </span>
              </div>
            </div>

            {/* Nav links */}
            <div className="col-xxl-5 col-xl-5 col-lg-5 d-none d-lg-block">
              <nav style={{ display: 'flex', gap: 36, justifyContent: 'center' }}>
                {['Features', 'Workflow', 'About'].map(label => (
                  <a key={label}
                    href={`#${label.toLowerCase()}`}
                    style={{
                      fontSize: 15, fontWeight: 500,
                      color: '#5F6368',
                      textDecoration: 'none',
                      fontFamily: 'var(--tp-ff-urban)',
                      transition: 'color 0.2s',
                    }}
                    onMouseEnter={e => e.currentTarget.style.color = '#dc2626'}
                    onMouseLeave={e => e.currentTarget.style.color = '#5F6368'}
                  >
                    {label}
                  </a>
                ))}
              </nav>
            </div>

            {/* CTA buttons */}
            <div className="col-xxl-4 col-xl-4 col-lg-4 col-md-8 col-6">
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 12 }}>
                <button
                  onClick={onStartDicom}
                  className="tp-btn-border tp-btn-hover d-none d-md-inline-block"
                  style={{ border: 'none', cursor: 'pointer', whiteSpace: 'nowrap' }}
                >
                  <span>DICOM / NIfTI</span>
                  <b></b>
                </button>
                <button
                  onClick={onOpenApp}
                  className="tp-btn-blue-sm tp-btn-hover alt-color-black"
                  style={{ border: 'none', cursor: 'pointer' }}
                >
                  <span>Open App</span>
                  <b></b>
                </button>
              </div>
            </div>

          </div>
        </div>
      </div>

      {/* ── HERO ─────────────────────────────────────────── */}
      <div
        className="p-relative"
        style={{
          minHeight: '100vh',
          display: 'flex', alignItems: 'center',
          backgroundImage: 'url(/image.png)',
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          backgroundRepeat: 'no-repeat',
          overflow: 'hidden',
          paddingTop: 120, paddingBottom: 80,
        }}
      >
        <div className="container" style={{ position: 'relative', zIndex: 3, width: '100%' }}>
          <div className="row">
            <div className="col-xl-7 col-lg-8">

              <div className="tp-hero-title-box">

                {/* Badge */}
                <motion.div
                  initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 0.05 }}
                  style={{
                    display: 'inline-flex', alignItems: 'center', gap: 8,
                    background: 'rgba(220,38,38,0.1)', border: '1px solid rgba(220,38,38,0.35)',
                    borderRadius: 100, padding: '6px 18px',
                    fontSize: 12, fontWeight: 600, color: '#dc2626',
                    marginBottom: 32, letterSpacing: '0.04em',
                    textTransform: 'uppercase', fontFamily: 'var(--tp-ff-urban)',
                  }}
                >
                  <span style={{ width: 7, height: 7, borderRadius: '50%', background: '#dc2626', flexShrink: 0 }} />
                  Explainable AI · Intracranial Aneurysm Research
                </motion.div>

                {/* Main heading */}
                <motion.h1
                  initial={{ opacity: 0, y: 28 }} animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.65, ease: [0.22, 1, 0.36, 1], delay: 0.12 }}
                  style={{
                    fontFamily: 'var(--tp-ff-urban)',
                    fontWeight: 800,
                    fontSize: 'clamp(32px, 4vw, 54px)',
                    lineHeight: 1.0,
                    letterSpacing: '-0.02em',
                    color: '#000229',
                    marginBottom: 24,
                  }}
                >
                  Predict Aneurysm{' '}<br />
                  <span style={{ color: '#dc2626', fontStyle: 'normal', fontWeight: 800, fontFamily: 'inherit' }}>Risk.</span>
                  <br />
                  Understand{' '}
                  <span style={{ color: '#dc2626', fontStyle: 'normal', fontWeight: 800, fontFamily: 'inherit' }}>Why.</span>
                </motion.h1>

                {/* Sub-heading */}
                <motion.p
                  initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: 0.22 }}
                  style={{ marginBottom: 0, color: '#3d3d3d' }}
                >
                  Upload a 3D mesh or DICOM scan. AneuXplain returns <br />a calibrated rupture
                  probability score and a per-vertex heatmap showing exactly which
                  geometry is driving the risk.
                </motion.p>

              </div>

              {/* CTA buttons */}
              <motion.div
                className="tp-hero-btn-3"
                initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.32 }}
                style={{ marginTop: 44 }}
              >
                <button
                  onClick={onStartMesh}
                  className="tp-btn-blue-lg tp-btn-hover alt-color-black"
                  style={{ border: 'none', cursor: 'pointer', margin: '0 8px 14px' }}
                >
                  <span>Upload Mesh</span>
                  <b></b>
                </button>
                <button
                  onClick={onStartDicom}
                  className="tp-btn-border tp-btn-hover alt-color-black"
                  style={{ border: 'none', cursor: 'pointer', margin: '0 8px 14px' }}
                >
                  <span>Upload DICOM / NIfTI</span>
                  <b></b>
                </button>
              </motion.div>

            </div>
          </div>
        </div>
      </div>

      {/* ── STATS / COUNTERS ─────────────────────────────── */}
      <div className="pb-80 pt-80" style={{ borderTop: '1px solid #EBECF0', borderBottom: '1px solid #EBECF0' }}>
        <div className="container">
          <div className="row gx-0 justify-content-center">
            {[
              { v: '6+',        l: 'Morphological Parameters' },
              { v: '<2s',       l: 'Time to Result'           },
              { v: 'XAI',       l: 'Fully Explainable'        },
              { v: 'OBJ/DICOM', l: 'Input Formats'            },
            ].map((item, i) => (
              <div key={i} className="col-xl-3 col-lg-3 col-md-6 col-6">
                <motion.div
                  {...inView(i * 0.08)}
                  style={{
                    textAlign: 'center',
                    padding: '40px 20px',
                    borderRight: i < 3 ? '1px solid #EBECF0' : 'none',
                  }}
                >
                  <div className="tp-counter-item">
                    <h4 style={{ color: '#dc2626' }}>{item.v}</h4>
                    <p>{item.l}</p>
                  </div>
                </motion.div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── FEATURES / SERVICES ──────────────────────────── */}
      <div id="features" className="pt-100 pb-80">
        <div className="container">

          {/* Section heading */}
          <div className="row mb-60">
            <div className="col-xl-12">
              <motion.div
                {...inView(0)}
                className="tp-service-section-wrapper d-flex justify-content-between align-items-end"
              >
                <h3 className="tp-section-title-3">
                  AI Analysis That <br /><span>Handles it All.</span>
                </h3>
                <button
                  onClick={onOpenApp}
                  className="tp-btn-blue-lg tp-btn-hover alt-color-black mb-10 d-none d-md-inline-block"
                  style={{ border: 'none', cursor: 'pointer' }}
                >
                  <span>Open Platform</span>
                  <b></b>
                </button>
              </motion.div>
            </div>
          </div>

          <div className="row align-items-stretch">

            {/* Large featured card */}
            <div className="col-xl-8 col-lg-7" style={{ display: 'flex', flexDirection: 'column' }}>
              <motion.div
                {...inView(0.05)}
                className="tp-service-3-item p-relative z-index"
                style={{
                  background: 'linear-gradient(135deg, #7f1d1d 0%, #991b1b 50%, #dc2626 100%)',
                  flex: 1,
                  marginBottom: 30,
                }}
              >
                <div className="tp-service-3-icon">
                  <svg width="58" height="58" viewBox="0 0 58 58" fill="none">
                    <circle cx="29" cy="29" r="22" stroke="white" strokeWidth="1.5" strokeOpacity="0.8"/>
                    <path d="M17 29 Q22 18 29 29 Q36 40 41 29" stroke="white" strokeWidth="2" fill="none" strokeLinecap="round"/>
                    <circle cx="29" cy="29" r="5" fill="white" fillOpacity="0.25"/>
                    <circle cx="29" cy="29" r="2" fill="white"/>
                  </svg>
                </div>
                <div className="tp-service-3-content">
                  <span>AI Risk Scoring</span>
                  <h4 className="tp-service-3-title-sm">
                    Calibrated Rupture Probability <br />from Geometric Deep Learning
                  </h4>
                  <p style={{
                    color: 'rgba(255,255,255,0.7)', fontSize: 15,
                    lineHeight: 1.75, marginTop: 16, marginBottom: 0,
                    maxWidth: 460,
                  }}>
                    AneuXplain analyses the 3D geometry of your aneurysm mesh and returns a
                    rupture probability score in under 2 seconds — fully explained through a
                    per-vertex heatmap and 6 clinically validated morphological parameters.
                    The underlying model achieves an AUC of 0.82 and was validated on a
                    cohort of 120+ patient scans, extracting features such as aspect ratio,
                    size ratio, undulation index, and ellipticity index to distinguish
                    ruptured from unruptured aneurysms with high specificity.
                  </p>
                </div>
                <div className="tp-service-3-btn" style={{ marginTop: 32 }}>
                  <button
                    onClick={onStartMesh}
                    className="tp-btn-white-solid"
                    style={{ border: 'none', cursor: 'pointer' }}
                  >
                    Try Now
                  </button>
                </div>
                {/* decorative rings */}
                <div style={{ position: 'absolute', right: -40, bottom: -40, opacity: 0.08, pointerEvents: 'none' }}>
                  <svg width="260" height="260" viewBox="0 0 260 260" fill="none">
                    <circle cx="130" cy="130" r="120" stroke="white" strokeWidth="1.5"/>
                    <circle cx="130" cy="130" r="80"  stroke="white" strokeWidth="1.5"/>
                    <circle cx="130" cy="130" r="40"  stroke="white" strokeWidth="1.5"/>
                  </svg>
                </div>
              </motion.div>
            </div>

            {/* Small cards column */}
            <div className="col-xl-4 col-lg-5">
              {/* Morphological Analysis */}
              <motion.div {...inView(0.1)} className="tp-service-sm-item mb-30 d-flex flex-column justify-content-between">
                <div className="tp-service-sm-icon">
                  <svg width="52" height="48" viewBox="0 0 52 48" fill="none">
                    <path d="M4 8h44M4 16h36M4 24h28M4 32h18M4 40h10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                    <circle cx="43" cy="36" r="8" stroke="currentColor" strokeWidth="1.5"/>
                    <path d="M43 32v4l2.5 2.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                  </svg>
                </div>
                <div className="tp-service-sm-content">
                  <span>Morphological Analysis</span>
                  <h3 className="tp-service-sm-title">
                    6 Clinically Validated Geometric Features.
                  </h3>
                  <div className="tp-service-sm-link">
                    <a href="#" onClick={e => { e.preventDefault(); onStartMesh(); }}>
                      Explore Analysis <ArrowRight style={{ width: 14, height: 14, display: 'inline', verticalAlign: 'middle', marginLeft: 4 }} />
                    </a>
                  </div>
                </div>
              </motion.div>

              {/* Heatmap */}
              <motion.div {...inView(0.15)} className="tp-service-sm-item d-flex flex-column justify-content-between">
                <div className="tp-service-sm-icon">
                  <svg width="52" height="52" viewBox="0 0 52 52" fill="none">
                    <rect x="4" y="4" width="44" height="44" rx="6" stroke="currentColor" strokeWidth="1.5"/>
                    <path d="M8 44l9-11 7 7 9-14 11 18" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </div>
                <div className="tp-service-sm-content">
                  <span>Explainability Heatmap</span>
                  <h3 className="tp-service-sm-title">
                    Per-Vertex Risk Sensitivity Overlay.
                  </h3>
                  <div className="tp-service-sm-link">
                    <a href="#" onClick={e => { e.preventDefault(); onStartMesh(); }}>
                      View Heatmap <ArrowRight style={{ width: 14, height: 14, display: 'inline', verticalAlign: 'middle', marginLeft: 4 }} />
                    </a>
                  </div>
                </div>
              </motion.div>
            </div>

          </div>

          {/* Bottom row of cards */}
          <div className="row" style={{ marginTop: 30 }}>
            {[
              {
                icon: (
                  <svg width="52" height="52" viewBox="0 0 52 52" fill="none">
                    <path d="M26 4L4 16v20l22 12 22-12V16L26 4z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round"/>
                    <path d="M26 4v44M4 16l22 12 22-12" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round"/>
                  </svg>
                ),
                label: '3D Visualization',
                title: 'Interactive OBJ & DICOM 3D Viewer.',
                link: () => onOpenApp(),
                linkText: 'Open Viewer',
              },
              {
                icon: (
                  <svg width="52" height="52" viewBox="0 0 52 52" fill="none">
                    <rect x="6" y="6" width="40" height="40" rx="8" stroke="currentColor" strokeWidth="1.5"/>
                    <path d="M16 26h20M26 16v20" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                    <circle cx="26" cy="26" r="6" stroke="currentColor" strokeWidth="1.5"/>
                  </svg>
                ),
                label: 'Counterfactual Generation',
                title: 'Generate Safer Aneurysm Shape Alternatives.',
                link: () => onOpenApp(),
                linkText: 'Explore',
              },
              {
                icon: (
                  <svg width="52" height="52" viewBox="0 0 52 52" fill="none">
                    <path d="M10 42V28M20 42V18M30 42V24M40 42V10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                    <path d="M6 42h40" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                  </svg>
                ),
                label: 'PDF Report Export',
                title: 'Full Clinical Report in One Click.',
                link: () => onOpenApp(),
                linkText: 'Learn More',
              },
            ].map((card, i) => (
              <div key={i} className="col-xl-4 col-lg-4 col-md-4">
                <motion.div {...inView(i * 0.08)} className="tp-service-sm-item mb-30 d-flex flex-column justify-content-between" style={{ minHeight: 360 }}>
                  <div className="tp-service-sm-icon">{card.icon}</div>
                  <div className="tp-service-sm-content">
                    <span>{card.label}</span>
                    <h3 className="tp-service-sm-title">{card.title}</h3>
                    <div className="tp-service-sm-link">
                      <a href="#" onClick={e => { e.preventDefault(); card.link(); }}>
                        {card.linkText} <ArrowRight style={{ width: 14, height: 14, display: 'inline', verticalAlign: 'middle', marginLeft: 4 }} />
                      </a>
                    </div>
                  </div>
                </motion.div>
              </div>
            ))}
          </div>

        </div>
      </div>

      {/* ── HOW IT WORKS ─────────────────────────────────── */}
      <div id="workflow" className="pt-100 pb-80" style={{ borderTop: '1px solid #EBECF0' }}>
        <div className="container">

          {/* Same heading pattern as Features section */}
          <div className="row mb-60">
            <div className="col-xl-12">
              <motion.div
                {...inView(0)}
                className="tp-service-section-wrapper d-flex justify-content-between align-items-end"
              >
                <h3 className="tp-section-title-3">
                  Scan to Insight<br /><span>in Three Steps.</span>
                </h3>
                <button
                  onClick={onOpenApp}
                  className="tp-btn-blue-lg tp-btn-hover alt-color-black mb-10 d-none d-md-inline-block"
                  style={{ border: 'none', cursor: 'pointer' }}
                >
                  <span>Try the Platform</span>
                  <b></b>
                </button>
              </motion.div>
            </div>
          </div>

          {/* Step cards — same structure as tp-service-sm-item cards */}
          <div className="row align-items-stretch">
            {[
              {
                n: '01',
                label: 'Upload',
                title: 'Upload Your Scan.',
                body: 'Drag and drop a 3D mesh (.obj/.ply/.stl) or a DICOM/NIfTI volume. AneuXplain automatically segments the cerebrovascular tree for volumetric data.',
                linkText: 'Upload Mesh',
                onClick: onStartMesh,
                icon: (
                  <svg width="52" height="52" viewBox="0 0 52 52" fill="none">
                    <path d="M26 4L4 14v24l22 10 22-10V14L26 4z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round"/>
                    <path d="M26 38V26M20 32l6-6 6 6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                ),
              },
              {
                n: '02',
                label: 'Analysis',
                title: 'AI Morphological Analysis.',
                body: 'Six clinically validated geometric features are extracted and scored by a gradient-boosted XAI classifier trained on real aneurysm cohorts.',
                linkText: 'See Features',
                onClick: onOpenApp,
                icon: (
                  <svg width="52" height="52" viewBox="0 0 52 52" fill="none">
                    <circle cx="26" cy="26" r="18" stroke="currentColor" strokeWidth="1.5"/>
                    <circle cx="26" cy="26" r="7" stroke="currentColor" strokeWidth="1.5"/>
                    <path d="M26 8v6M26 38v6M8 26h6M38 26h6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                  </svg>
                ),
              },
              {
                n: '03',
                label: 'Export',
                title: 'Review & Export.',
                body: 'Interact with the 3D viewer, toggle the heatmap overlay, read clinical significance for each parameter, and export a full PDF report in one click.',
                linkText: 'Open Viewer',
                onClick: onOpenApp,
                icon: (
                  <svg width="52" height="52" viewBox="0 0 52 52" fill="none">
                    <rect x="10" y="6" width="32" height="40" rx="3" stroke="currentColor" strokeWidth="1.5"/>
                    <path d="M18 18h16M18 26h16M18 34h10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                    <path d="M34 32l6 6M34 38l6-6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                  </svg>
                ),
              },
            ].map((step, i) => (
              <div key={i} className="col-xl-4 col-lg-4 col-md-4" style={{ display: 'flex', flexDirection: 'column' }}>
                <motion.div
                  {...inView(i * 0.1)}
                  className="tp-service-sm-item mb-30 d-flex flex-column justify-content-between"
                  style={{ flex: 1, minHeight: 340 }}
                >
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 28 }}>
                      <div className="tp-service-sm-icon" style={{ margin: 0 }}>
                        {step.icon}
                      </div>
                      <span style={{
                        fontSize: 48, fontWeight: 800, color: '#F0F1F5',
                        fontFamily: 'var(--tp-ff-urban)', lineHeight: 1, letterSpacing: '-0.04em',
                        userSelect: 'none',
                      }}>
                        {step.n}
                      </span>
                    </div>
                    <div className="tp-service-sm-content">
                      <span>{step.label}</span>
                      <h3 className="tp-service-sm-title">{step.title}</h3>
                      <p style={{ fontSize: 15, color: '#5F6368', lineHeight: 1.8, marginBottom: 16 }}>
                        {step.body}
                      </p>
                    </div>
                  </div>
                  <div className="tp-service-sm-link">
                    <a href="#" onClick={e => { e.preventDefault(); step.onClick(); }}>
                      {step.linkText} <ArrowRight style={{ width: 14, height: 14, display: 'inline', verticalAlign: 'middle', marginLeft: 4 }} />
                    </a>
                  </div>
                </motion.div>
              </div>
            ))}
          </div>

        </div>
      </div>

      {/* ── CTA BANNER ───────────────────────────────────── */}
      <div id="about" className="pt-100 pb-100">
        <div className="container">
          <motion.div
            {...inView(0)}
            style={{
              borderRadius: 28,
              background: 'linear-gradient(135deg, #7f1d1d 0%, #991b1b 45%, #dc2626 100%)',
              padding: '72px 80px',
              display: 'flex', alignItems: 'center',
              justifyContent: 'space-between', gap: 40,
              position: 'relative', overflow: 'hidden',
            }}
          >
            {/* bg circles decoration */}
            <div style={{ position: 'absolute', right: -80, top: -80, opacity: 0.06, pointerEvents: 'none' }}>
              <svg width="380" height="380" viewBox="0 0 380 380" fill="none">
                <circle cx="190" cy="190" r="170" stroke="white" strokeWidth="2"/>
                <circle cx="190" cy="190" r="110" stroke="white" strokeWidth="2"/>
                <circle cx="190" cy="190" r="55" stroke="white" strokeWidth="2"/>
              </svg>
            </div>

            <div style={{ color: '#fff', position: 'relative', zIndex: 2 }}>
              <h2 style={{
                fontSize: 38, fontWeight: 700, color: '#fff',
                marginBottom: 14, lineHeight: 1.2, letterSpacing: '-0.02em',
                fontFamily: 'var(--tp-ff-urban)',
              }}>
                Ready to analyze <br />your first scan?
              </h2>
              <p style={{ fontSize: 16, color: 'rgba(255,255,255,0.72)', lineHeight: 1.65, marginBottom: 0 }}>
                No account. No setup. Upload and get results in under 2 seconds.
              </p>
            </div>

            <div style={{ display: 'flex', gap: 14, flexShrink: 0, position: 'relative', zIndex: 2, flexWrap: 'wrap' }}>
              <button
                onClick={onStartMesh}
                className="tp-btn-blue-lg tp-btn-hover"
                style={{
                  border: 'none', cursor: 'pointer',
                  background: '#fff', color: '#000229',
                  boxShadow: '0 4px 20px rgba(0,0,0,0.18)',
                }}
              >
                <span style={{ color: '#000229' }}>Upload Mesh</span>
                <b style={{ background: 'rgba(0,0,0,0.06)' }}></b>
              </button>
              <button
                onClick={onStartDicom}
                style={{
                  height: 60, lineHeight: '60px', padding: '0 35px',
                  borderRadius: 100, background: 'rgba(255,255,255,0.12)',
                  border: '1px solid rgba(255,255,255,0.35)',
                  color: '#fff', fontSize: 16, fontWeight: 700,
                  cursor: 'pointer', fontFamily: 'var(--tp-ff-urban)',
                  letterSpacing: '0.02em', transition: 'background 0.3s',
                }}
                onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,255,255,0.22)'}
                onMouseLeave={e => e.currentTarget.style.background = 'rgba(255,255,255,0.12)'}
              >
                DICOM / NIfTI
              </button>
            </div>
          </motion.div>
        </div>
      </div>

      {/* ── FOOTER ───────────────────────────────────────── */}
      <footer style={{ borderTop: '1px solid #EBECF0', padding: '28px 0' }}>
        <div className="container">
          <div className="d-flex align-items-center justify-content-between">
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <img src="/logoAneuX.png" alt="AneuXplain"
                style={{ width: 28, height: 28, borderRadius: 8, objectFit: 'contain' }} />
              <span style={{ fontSize: 14, color: '#5F6368', fontWeight: 600, fontFamily: 'var(--tp-ff-urban)' }}>
                AneuXplain
              </span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <Shield style={{ width: 12, height: 12, color: '#9A9DA7' }} />
              <span style={{ fontSize: 11, color: '#9A9DA7' }}>
                Research prototype · not for clinical diagnosis · data processed locally
              </span>
            </div>
          </div>
        </div>
      </footer>

    </div>
  );
}
