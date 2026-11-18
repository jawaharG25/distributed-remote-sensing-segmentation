const runs = [
  ['LoveDA','DeepLabV3+','1','Baseline','0.0341','0.5018','3,967.20'],['LoveDA','DeepLabV3+','2','SyncBatchNorm','0.0272','0.4897','2,938.84'],['LoveDA','DeepLabV3+','4','SyncBatchNorm','0.0253','0.4836','1,536.19'],['LoveDA','DeepLabV3+','2','LocalBatchNorm','0.0328','0.4957','2,237.59'],['LoveDA','DeepLabV3+','4','LocalBatchNorm','0.0320','0.4905','1,127.97'],['LoveDA','SegFormer-B2','1','Baseline','0.0584','0.5241','9,347.62'],['LoveDA','SegFormer-B2','2','SyncBatchNorm','0.0503','0.5244','4,878.99'],['LoveDA','SegFormer-B2','4','SyncBatchNorm','0.0465','0.5186','2,452.44'],['LoveDA','SegFormer-B2','2','LocalBatchNorm','0.0528','0.5248','4,878.33'],['LoveDA','SegFormer-B2','4','LocalBatchNorm','0.0503','0.5244','2,448.10'],['LandCover.ai','DeepLabV3+','1','Baseline','0.0080','0.8358','3,145.06'],['LandCover.ai','DeepLabV3+','2','SyncBatchNorm','0.0069','0.8355','1,749.18'],['LandCover.ai','DeepLabV3+','4','SyncBatchNorm','0.0074','0.8346','831.44'],['LandCover.ai','DeepLabV3+','2','LocalBatchNorm','0.0084','0.8357','1,666.82'],['LandCover.ai','DeepLabV3+','4','LocalBatchNorm','0.0107','0.8324','736.37'],['LandCover.ai','SegFormer-B2','1','Baseline','0.0095','0.8609','4,380.67'],['LandCover.ai','SegFormer-B2','2','SyncBatchNorm','0.0085','0.8599','2,292.81'],['LandCover.ai','SegFormer-B2','4','SyncBatchNorm','0.0085','0.8604','1,123.08'],['LandCover.ai','SegFormer-B2','2','LocalBatchNorm','0.0099','0.8611','2,288.23'],['LandCover.ai','SegFormer-B2','4','LocalBatchNorm','0.0103','0.8586','1,122.16']
];

const tbody = document.querySelector('#run-rows');
const modeClass = (mode) => mode === 'SyncBatchNorm' ? 'bn-sync' : mode === 'LocalBatchNorm' ? 'bn-local' : 'bn-base';
function renderRows(filter = 'all') {
  tbody.innerHTML = runs.filter((run) => filter === 'all' || run[0] === filter).map((run) => `<tr><td class="dataset-name">${run[0]}</td><td class="arch">${run[1]}</td><td class="gpu">${run[2]} GPU</td><td><span class="bn ${modeClass(run[3])}">${run[3]}</span></td><td>${run[4]}</td><td class="miou">${run[5]}</td><td class="time">${run[6]} s</td></tr>`).join('');
}
renderRows();
document.querySelectorAll('.filter').forEach((button) => button.addEventListener('click', () => {
  document.querySelectorAll('.filter').forEach((item) => item.classList.remove('active'));
  button.classList.add('active');
  renderRows(button.dataset.dataset);
}));

const menu = document.querySelector('.menu-toggle');
const nav = document.querySelector('.nav');
menu.addEventListener('click', () => {
  const open = nav.classList.toggle('open');
  menu.setAttribute('aria-expanded', String(open));
  menu.textContent = open ? 'Close' : 'Menu';
});
nav.querySelectorAll('a').forEach((link) => link.addEventListener('click', () => {
  nav.classList.remove('open');
  menu.setAttribute('aria-expanded', 'false');
  menu.textContent = 'Menu';
}));

const observer = new IntersectionObserver((entries) => entries.forEach((entry) => {
  if (entry.isIntersecting) {
    entry.target.classList.add('visible');
    observer.unobserve(entry.target);
  }
}), { threshold: 0.12 });
document.querySelectorAll('.reveal').forEach((element) => observer.observe(element));
