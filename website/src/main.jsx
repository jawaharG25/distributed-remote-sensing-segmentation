import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';

const GITHUB_URL = 'https://github.com/jawaharG25/distributed-remote-sensing-segmentation';

const runs = [
  ['LoveDA', 'DeepLabV3+', '1', 'Baseline', '0.5018', '3,967'],
  ['LoveDA', 'DeepLabV3+', '2', 'SyncBatchNorm', '0.4897', '2,939'],
  ['LoveDA', 'DeepLabV3+', '4', 'SyncBatchNorm', '0.4836', '1,536'],
  ['LoveDA', 'DeepLabV3+', '2', 'LocalBatchNorm', '0.4957', '2,238'],
  ['LoveDA', 'DeepLabV3+', '4', 'LocalBatchNorm', '0.4905', '1,128'],
  ['LoveDA', 'SegFormer-B2', '1', 'Baseline', '0.5241', '9,348'],
  ['LoveDA', 'SegFormer-B2', '2', 'SyncBatchNorm', '0.5244', '4,879'],
  ['LoveDA', 'SegFormer-B2', '4', 'SyncBatchNorm', '0.5186', '2,452'],
  ['LoveDA', 'SegFormer-B2', '2', 'LocalBatchNorm', '0.5248', '4,878'],
  ['LoveDA', 'SegFormer-B2', '4', 'LocalBatchNorm', '0.5244', '2,448'],
  ['LandCover.ai', 'DeepLabV3+', '1', 'Baseline', '0.8358', '3,145'],
  ['LandCover.ai', 'DeepLabV3+', '2', 'SyncBatchNorm', '0.8355', '1,749'],
  ['LandCover.ai', 'DeepLabV3+', '4', 'SyncBatchNorm', '0.8346', '831'],
  ['LandCover.ai', 'DeepLabV3+', '2', 'LocalBatchNorm', '0.8357', '1,667'],
  ['LandCover.ai', 'DeepLabV3+', '4', 'LocalBatchNorm', '0.8324', '736'],
  ['LandCover.ai', 'SegFormer-B2', '1', 'Baseline', '0.8609', '4,381'],
  ['LandCover.ai', 'SegFormer-B2', '2', 'SyncBatchNorm', '0.8599', '2,293'],
  ['LandCover.ai', 'SegFormer-B2', '4', 'SyncBatchNorm', '0.8604', '1,123'],
  ['LandCover.ai', 'SegFormer-B2', '2', 'LocalBatchNorm', '0.8611', '2,288'],
  ['LandCover.ai', 'SegFormer-B2', '4', 'LocalBatchNorm', '0.8586', '1,122']
];

const scaling = [
  { label: 'LoveDA · DeepLabV3+', values: [3967, 2939, 1536], speedup: '2.58×' },
  { label: 'LoveDA · SegFormer-B2', values: [9348, 4879, 2452], speedup: '3.81×' },
  { label: 'LandCover.ai · DeepLabV3+', values: [3145, 1749, 831], speedup: '3.78×' },
  { label: 'LandCover.ai · SegFormer-B2', values: [4381, 2293, 1123], speedup: '3.90×' }
];

function SectionTitle({ number, eyebrow, title, children }) {
  return (
    <div className="section-title">
      <div className="eyebrow"><span>{number}</span>{eyebrow}</div>
      <h2>{title}</h2>
      {children && <p>{children}</p>}
    </div>
  );
}

function ArchitectureDiagram() {
  return (
    <div className="architecture-diagram" aria-label="Distributed training architecture">
      <div className="data-node"><span>512 × 512</span><strong>Image + mask</strong></div>
      <div className="connector">→</div>
      <div className="gpu-cluster">
        <div className="cluster-label">PyTorch DDP · NCCL</div>
        <div className="gpu-row">{['GPU 0', 'GPU 1', 'GPU 2', 'GPU 3'].map((gpu) => <div className="gpu" key={gpu}><span>{gpu}</span><small>forward</small><small>backward</small></div>)}</div>
        <div className="sync-bar">all-reduce metrics / gradients</div>
      </div>
      <div className="connector">→</div>
      <div className="data-node output"><span>mIoU</span><strong>Validation</strong></div>
    </div>
  );
}

function BarChart() {
  const max = 9348;
  return (
    <div className="bar-chart">
      {scaling.map((item) => (
        <div className="bar-group" key={item.label}>
          <div className="bar-label"><span>{item.label}</span><b>{item.speedup} at 4 GPUs</b></div>
          <div className="bars">
            {item.values.map((value, index) => <div className="bar-wrap" key={value}><div className="bar" style={{ height: `${Math.max(7, value / max * 100)}%` }}><span>{value.toLocaleString()}s</span></div><small>{[1, 2, 4][index]} GPU</small></div>)}
          </div>
        </div>
      ))}
    </div>
  );
}

function App() {
  const [active, setActive] = useState('all');
  const [menuOpen, setMenuOpen] = useState(false);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    setVisible(true);
    const onScroll = () => setVisible(window.scrollY > 24);
    window.addEventListener('scroll', onScroll);
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  const filteredRuns = runs.filter((run) => active === 'all' || run[0] === active);
  const nav = ['abstract', 'datasets', 'method', 'results', 'discussion', 'figures', 'artifacts'];

  return (
    <>
      <header className={`site-nav ${visible ? 'scrolled' : ''}`}>
        <a className="wordmark" href="#top"><span>DS</span><strong>Distributed<br />Segmentation</strong></a>
        <button className="menu-button" onClick={() => setMenuOpen(!menuOpen)} aria-expanded={menuOpen}>Menu</button>
        <nav className={menuOpen ? 'open' : ''}>{nav.map((item) => <a key={item} href={`#${item}`} onClick={() => setMenuOpen(false)}>{item}</a>)}<a className="github-link" href={GITHUB_URL} target="_blank" rel="noreferrer">GitHub ↗</a></nav>
      </header>

      <main id="top">
        <section className="hero page-width">
          <div className="hero-copy">
            <p className="eyebrow">Master's thesis · Computer Science</p>
            <h1>Distributed semantic segmentation for <em>remote-sensing imagery.</em></h1>
            <p className="lead">An experimental study of GPU scaling, BatchNorm synchronization, and model architecture using PyTorch Distributed Data Parallel.</p>
            <div className="hero-actions"><a className="button primary" href="#abstract">Read the thesis <span>↓</span></a><a className="button secondary" href={GITHUB_URL} target="_blank" rel="noreferrer">View GitHub ↗</a></div>
            <div className="hero-facts"><div><b>20</b><span>recorded runs</span></div><div><b>2</b><span>datasets</span></div><div><b>1–4</b><span>GPUs</span></div><div><b>50</b><span>epochs</span></div></div>
          </div>
          <div className="hero-figure"><div className="hero-grid"></div><div className="hero-label label-a">INPUT<br /><b>REMOTE SENSING</b></div><div className="hero-label label-b">MODEL<br /><b>DDP × 4</b></div><div className="hero-label label-c">OUTPUT<br /><b>mIoU</b></div><div className="hero-network"><div className="network-core">DDP<small>global sync</small></div>{['01', '02', '03', '04'].map((node) => <span className={`network-node n-${node}`} key={node}>{node}</span>)}</div></div>
        </section>

        <section id="abstract" className="page-width section">
          <SectionTitle number="01" eyebrow="Abstract" title="The research in one page.">This thesis evaluates how distributed hardware and synchronized normalization affect semantic segmentation quality and training time for high-resolution remote-sensing imagery.</SectionTitle>
          <div className="abstract-grid"><div className="abstract-text"><p>Two architectures—DeepLabV3+ with a ResNet-50 encoder and SegFormer-B2—are trained on LoveDA and LandCover.ai. The experiments use a fixed 50-epoch recipe with focal loss, AdamW, warm-up, and cosine learning-rate decay.</p><p>PyTorch DDP scales training from one to four GPUs. Multi-GPU runs are then split into synchronized BatchNorm and local BatchNorm conditions to isolate the effect of per-GPU micro-batch statistics.</p></div><div className="finding-box"><span>Key finding</span><strong>SegFormer-B2 reaches 0.8609 mIoU on LandCover.ai at one GPU, while four-GPU training reaches up to 3.90× speedup.</strong><small>Observed results from the recorded experiment logs.</small></div></div>
        </section>

        <section id="datasets" className="tint-section"><div className="page-width section"><SectionTitle number="02" eyebrow="Datasets & architectures" title="The experimental space.">The comparison uses two label spaces and two different inductive biases while keeping preprocessing and optimization consistent.</SectionTitle><div className="dataset-cards"><article><div className="dataset-image love-image"><span>LoveDA</span></div><h3>LoveDA</h3><p>Urban and rural scenes, resized from 1024×1024 to 512×512.</p><dl><div><dt>Classes</dt><dd>8</dd></div><div><dt>Ignore index</dt><dd>0</dd></div><div><dt>Input</dt><dd>512²</dd></div></dl></article><article><div className="dataset-image land-image"><span>LandCover.ai</span></div><h3>LandCover.ai</h3><p>Large map rasters statically tiled into 512×512 patches.</p><dl><div><dt>Classes</dt><dd>5</dd></div><div><dt>Ignore index</dt><dd>255</dd></div><div><dt>Input</dt><dd>512²</dd></div></dl></article></div><div className="architecture-cards"><article><span className="architecture-tag cnn">CNN</span><h3>DeepLabV3+ / ResNet-50</h3><p>ImageNet-pretrained encoder with atrous spatial context and a decoder for dense prediction.</p></article><article><span className="architecture-tag transformer">TRANSFORMER</span><h3>SegFormer-B2</h3><p>Hierarchical Transformer encoder initialized from <code>nvidia/mit-b2</code> with a lightweight decoder.</p></article></div></div></section>

        <section id="method" className="page-width section"><SectionTitle number="03" eyebrow="Methodology" title="A fixed recipe across changing systems.">Every run follows the same data preparation, optimization schedule, and metric definition. Only architecture, GPU count, and BatchNorm mode change.</SectionTitle><ArchitectureDiagram /><div className="method-grid"><article><span>DATA</span><b>ImageNet normalization</b><p>Images use bilinear resizing; masks use nearest-neighbor resizing to preserve class IDs.</p></article><article><span>LOSS</span><b>Focal Loss · γ = 2.0</b><p>Ignore labels are excluded from loss and mean IoU calculations.</p></article><article><span>OPTIMIZER</span><b>AdamW · batch 8/GPU</b><p>ResNet LR is 1e-4 × world size; SegFormer LR is 6e-5 × world size.</p></article><article><span>SCHEDULE</span><b>Warm-up + cosine decay</b><p>Five linear warm-up epochs followed by cosine learning-rate decay.</p></article></div></section>

        <section id="results" className="dark-section"><div className="page-width section"><SectionTitle number="04" eyebrow="Results" title="Quality and time tell different stories.">The complete recorded matrix is shown below. Times are sums of logged epoch times; final epochs also write predictions and checkpoints.</SectionTitle><div className="stat-row"><div><span>Best mIoU</span><b>0.8609</b><small>LandCover.ai · SegFormer-B2</small></div><div><span>Best speedup</span><b>3.90×</b><small>LandCover.ai · 4 GPUs</small></div><div><span>Best efficiency</span><b>97.88%</b><small>LandCover.ai · 4 GPUs</small></div></div><div className="chart-card"><div className="chart-heading"><h3>Scaling time</h3><span>Lower is faster · recorded seconds</span></div><BarChart /></div><div className="table-controls"><div>{['all', 'LoveDA', 'LandCover.ai'].map((filter) => <button className={active === filter ? 'active' : ''} onClick={() => setActive(filter)} key={filter}>{filter === 'all' ? 'All runs' : filter}</button>)}</div><span>{filteredRuns.length} configurations</span></div><div className="results-table"><table><thead><tr><th>Dataset</th><th>Architecture</th><th>GPU</th><th>BN</th><th>mIoU</th><th>50-epoch time</th></tr></thead><tbody>{filteredRuns.map((run) => <tr key={run.join('-')}><td>{run[0]}</td><td>{run[1]}</td><td>{run[2]}</td><td><span className={`badge ${run[3] === 'SyncBatchNorm' ? 'sync' : run[3] === 'LocalBatchNorm' ? 'local' : 'base'}`}>{run[3]}</span></td><td className="emphasis">{run[4]}</td><td>{run[5]} s</td></tr>)}</tbody></table></div></div></section>

        <section id="discussion" className="page-width section"><SectionTitle number="05" eyebrow="Discussion" title="What the evidence suggests.">The results are interpreted as observed behavior of this experiment set, not as a universal ranking of models or hardware.</SectionTitle><div className="discussion-grid"><article><span>01</span><h3>SegFormer has the quality advantage</h3><p>At one GPU, SegFormer exceeds DeepLabV3+ by 2.23 percentage points on LoveDA and 2.51 points on LandCover.ai.</p></article><article><span>02</span><h3>Scaling depends on the model</h3><p>SegFormer reaches 3.81× and 3.90× speedup at four GPUs, while DeepLabV3+ reaches 2.58× and 3.78×.</p></article><article><span>03</span><h3>SyncBatchNorm is not always better</h3><p>The effect is small and dataset-dependent. Local BatchNorm is higher for several LoveDA conditions, while Sync is higher for some LandCover.ai runs.</p></article></div><div className="class-panel"><div><h3>Hardest final classes</h3><p>Class-wise IoU shows where mean scores hide difficulty.</p></div><div className="class-bars"><label><span>LoveDA · Barren</span><i style={{ width: '25.25%' }}></i><b>0.2525</b></label><label><span>LoveDA · Forest</span><i style={{ width: '40.84%' }}></i><b>0.4084</b></label><label><span>LandCover.ai · Road</span><i style={{ width: '68.97%' }}></i><b>0.6897</b></label><label><span>LandCover.ai · Water</span><i style={{ width: '95.05%' }}></i><b>0.9505</b></label></div></div></section>

        <section id="figures" className="page-width section figures-section"><SectionTitle number="06" eyebrow="Figures" title="Visual outputs from the experiments.">The repository preserves representative GPU comparison collages for both architectures and datasets.</SectionTitle><div className="figure-grid">{[['segformer_landcover_gpu_collage.png','LandCover.ai · SegFormer-B2'],['deeplabv3plus_landcover_gpu_collage.png','LandCover.ai · DeepLabV3+'],['segformer_gpu_collage.png','LoveDA · SegFormer-B2'],['deeplabv3plus_gpu_collage.png','LoveDA · DeepLabV3+']].map(([file, label]) => <figure key={file}><img src={`${import.meta.env.BASE_URL}figures/${file}`} alt={label} /><figcaption>{label}</figcaption></figure>)}</div></section>

        <section id="artifacts" className="tint-section"><div className="page-width section"><SectionTitle number="07" eyebrow="Reproducibility" title="The website is only the summary.">The repository contains the source scripts, SLURM launch files, normalized logs, machine-readable tables, figures, and documentation used to produce this page.</SectionTitle><div className="artifact-grid"><a href={`${GITHUB_URL}/tree/main/src`} target="_blank" rel="noreferrer"><span>01</span><b>Training scripts</b><small>src/{'{dataset}'}/</small></a><a href={`${GITHUB_URL}/tree/main/slurm`} target="_blank" rel="noreferrer"><span>02</span><b>HPC launch files</b><small>slurm/{'{dataset}'}/</small></a><a href={`${GITHUB_URL}/tree/main/results/logs`} target="_blank" rel="noreferrer"><span>03</span><b>Recorded logs</b><small>results/logs/</small></a><a href={`${GITHUB_URL}/tree/main/results/tables`} target="_blank" rel="noreferrer"><span>04</span><b>Result tables</b><small>results/tables/*.csv</small></a></div><div className="limitations"><h3>Scope and limitations</h3><p>One logged run is available per configuration. No CPU-only or eight-GPU log is present, hardware allocations differ between jobs, and no repeated seeds or confidence intervals are available.</p></div><a className="button primary repo-button" href={GITHUB_URL} target="_blank" rel="noreferrer">Open the GitHub repository ↗</a></div></section>
      </main>
      <footer className="footer page-width"><span>Distributed semantic segmentation · Master's thesis</span><span>PyTorch DDP · LoveDA · LandCover.ai</span></footer>
    </>
  );
}

createRoot(document.getElementById('root')).render(<App />);
