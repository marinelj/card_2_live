// Drive room.html: read frame specs (JSON lines), render each, save JPEGs.
const { chromium } = require('/opt/node-tools/node_modules/playwright');
const fs = require('fs');
const [specFile, outDir] = process.argv.slice(2);
(async () => {
  const b = await chromium.launch({ args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const p = await b.newPage({ viewport: { width: 720, height: 1280 } });
  p.on('console', m => console.log('page:', m.text())); p.on('pageerror', e => console.log('err:', e.message));
  await p.goto('http://127.0.0.1:8765/room.html');
  await p.waitForFunction(() => window.ready === true, null, { timeout: 120000 });
  fs.mkdirSync(outDir, { recursive: true });
  const specs = fs.readFileSync(specFile, 'utf8').trim().split('\n').map(JSON.parse);
  const t0 = Date.now();
  for (let i = 0; i < specs.length; i++) {
    await p.evaluate(s => window.frame(s), specs[i]);
    await (await p.$('#out')).screenshot({ path: `${outDir}/${String(i + 1).padStart(3, '0')}.jpg`, type: 'jpeg', quality: 95 });
    if (i % 10 == 0) console.log('frame', i, ((Date.now() - t0) / 1000).toFixed(1) + 's');
  }
  await b.close();
})();
