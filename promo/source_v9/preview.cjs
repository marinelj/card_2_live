const { chromium } = require('/opt/node-tools/node_modules/playwright');
const fs = require('fs');
(async () => {
  const b = await chromium.launch({ args: ['--allow-file-access-from-files'] });
  const p = await b.newPage({ viewport: { width: 1920, height: 1080 } });
  const errs = [];
  p.on('pageerror', e => errs.push(e.message)); p.on('console', m => { if (m.type() === 'error') errs.push(m.text()); });
  await p.goto('file://' + __dirname + '/scene.html');
  await p.evaluate(() => window.READY);
  const times = process.argv.slice(2).map(Number);
  for (const t of times) {
    const d = await p.evaluate(t => { render(t); return document.getElementById('c').toDataURL('image/jpeg', 0.85); }, t);
    fs.writeFileSync(`prev_${t}.jpg`, Buffer.from(d.split(',')[1], 'base64'));
  }
  console.log('errors:', errs);
  await b.close();
})();
