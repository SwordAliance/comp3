const grid = document.getElementById('grid')
const pic = document.getElementById('pic')
const emp = document.getElementById('emp')
const cwrap = document.getElementById('cwrap')
const overlay = document.getElementById('overlay')
const dim = document.getElementById('dim')
const pring = document.getElementById('pring')
const rfill = document.getElementById('rfill')
const rpct = document.getElementById('rpct')
const CIRC = 2 * Math.PI * 34
const bbar = document.getElementById('bbar')
const analyzeBtn = document.getElementById('analyzeBtn')
const exportBtn = document.getElementById('exportBtn')
const results = document.getElementById('results')
const defectList = document.getElementById('defectList')
const ntf = document.getElementById('ntf')
const stageLabel = document.getElementById('stageLabel')
const pipelineStages = document.getElementById('pipelineStages')

let sel = new Set()
let cur = null
let curSrc = null
let allPhotos = []
let defects = []
let analyzing = false
let currentJobId = null

const SEV_COLORS = {
  'мелкий':      'rgba(204,170,80,.75)',
  'средний':     'rgba(200,120,50,.75)',
  'критический': 'rgba(190,60,55,.75)'
}

function notify(t) {
  ntf.textContent = t
  ntf.classList.add('pop')
  setTimeout(() => ntf.classList.remove('pop'), 1800)
}

/* ---------- Gallery ---------- */

async function fetchPhotos() {
  try {
    let res = await fetch('/api/photos')
    allPhotos = await res.json()
  } catch {
    allPhotos = []
  }
  return allPhotos
}

async function render() {
  await fetchPhotos()
  grid.innerHTML = ''

  if (!allPhotos.length) {
    grid.innerHTML = '<div class="nope">пока пусто —<br>сделайте фото на вкладке Съёмка</div>'
    return
  }

  allPhotos.forEach(ph => {
    let d = document.createElement('div')
    d.className = 'th' + (sel.has(ph.id) ? ' picked' : '')

    let img = document.createElement('img')
    img.src = ph.d
    d.appendChild(img)

    d.onclick = e => {
      if (e.shiftKey || e.ctrlKey || e.metaKey) {
        sel.has(ph.id) ? sel.delete(ph.id) : sel.add(ph.id)
        render()
      } else {
        openPhoto(ph)
      }
    }

    d.oncontextmenu = e => {
      e.preventDefault()
      sel.has(ph.id) ? sel.delete(ph.id) : sel.add(ph.id)
      render()
    }

    grid.appendChild(d)
  })
}

function openPhoto(ph) {
  cur = ph.id
  curSrc = ph.d
  rotDeg = 0
  pic.style.transition = 'none'
  pic.style.transform = 'none'
  pic.src = ph.d
  cwrap.style.display = ''
  emp.style.display = 'none'
  bbar.style.display = ''

  clearResults()

  sel.clear()
  sel.add(ph.id)
  renderGrid()

  pic.onload = () => {
    overlay.width = pic.naturalWidth
    overlay.height = pic.naturalHeight
  }
}

function renderGrid() {
  grid.innerHTML = ''
  allPhotos.forEach(ph => {
    let d = document.createElement('div')
    d.className = 'th' + (sel.has(ph.id) ? ' picked' : '')

    let img = document.createElement('img')
    img.src = ph.d
    d.appendChild(img)

    d.onclick = e => {
      if (e.shiftKey || e.ctrlKey || e.metaKey) {
        sel.has(ph.id) ? sel.delete(ph.id) : sel.add(ph.id)
        renderGrid()
      } else {
        openPhoto(ph)
      }
    }

    d.oncontextmenu = e => {
      e.preventDefault()
      sel.has(ph.id) ? sel.delete(ph.id) : sel.add(ph.id)
      renderGrid()
    }

    grid.appendChild(d)
  })
}

function clearResults() {
  defects = []
  currentJobId = null
  results.style.display = 'none'
  defectList.innerHTML = ''
  exportBtn.style.display = 'none'
  pipelineStages.style.display = 'none'
  dim.classList.remove('active')
  pring.classList.remove('active')
  stageLabel.textContent = ''
  stageLabel.style.display = 'none'
  rfill.style.strokeDashoffset = CIRC
  rpct.textContent = '0%'

  let ctx = overlay.getContext('2d')
  ctx.clearRect(0, 0, overlay.width, overlay.height)

  // Clear stage cards
  document.querySelectorAll('.stage-card img').forEach(img => { img.src = '' })
  document.querySelectorAll('.stage-card').forEach(c => c.classList.remove('active'))
}

/* ---------- Real ML Analysis ---------- */

analyzeBtn.onclick = async () => {
  if (!cur || analyzing) return
  analyzing = true
  analyzeBtn.disabled = true
  analyzeBtn.textContent = 'Анализ...'

  clearResults()

  dim.classList.add('active')
  pring.classList.add('active')
  stageLabel.style.display = ''
  stageLabel.textContent = 'Запуск...'
  rfill.style.strokeDashoffset = CIRC
  rpct.textContent = '0%'

  try {
    // Start analysis
    let resp = await fetch('/api/analyze/' + cur, { method: 'POST' })
    if (!resp.ok) throw new Error('Ошибка запуска анализа')
    let data = await resp.json()
    let jobId = data.job_id
    currentJobId = jobId

    // Poll for result
    let result = await pollForResult(jobId)

    // Show pipeline stages
    showPipelineStages(jobId, result)

    // Map defects (pass crop offset so coords map to full image)
    defects = mapDefects(result.defects || [], result.crop_offset || [0, 0])

    // Draw boxes and show results
    drawBoxes()
    showResults()

  } catch (e) {
    notify('Ошибка: ' + e.message)
  } finally {
    dim.classList.remove('active')
    pring.classList.remove('active')
    stageLabel.style.display = 'none'
    rfill.style.strokeDashoffset = CIRC
    rpct.textContent = '0%'
    analyzing = false
    analyzeBtn.disabled = false
    analyzeBtn.textContent = 'Запустить анализ'
  }
}

function pollForResult(jobId) {
  return new Promise((resolve, reject) => {
    let progress = 0
    let tick = setInterval(() => {
      let remaining = 90 - progress
      progress += remaining * 0.03
      let pct = Math.round(progress)
      rfill.style.strokeDashoffset = CIRC * (1 - progress / 100)
      rpct.textContent = pct + '%'
    }, 100)

    let poll = setInterval(async () => {
      try {
        let resp = await fetch('/api/result/' + jobId)
        let data = await resp.json()

        if (data.status === 'processing') {
          if (data.progress) {
            stageLabel.textContent = data.progress
          }
        } else if (data.status === 'done') {
          clearInterval(poll)
          clearInterval(tick)
          rfill.style.strokeDashoffset = 0
          rpct.textContent = '100%'
          // Brief pause to show 100%
          setTimeout(() => resolve(data.result), 300)
        } else if (data.status === 'error') {
          clearInterval(poll)
          clearInterval(tick)
          reject(new Error(data.error || 'Ошибка анализа'))
        }
      } catch (e) {
        clearInterval(poll)
        clearInterval(tick)
        reject(new Error('Потеряна связь с сервером'))
      }
    }, 1000)
  })
}

function showPipelineStages(jobId, result) {
  let base = '/api/result/' + jobId + '/file/'

  let stageOriginal = document.getElementById('stageOriginal')
  let stageSegmented = document.getElementById('stageSegmented')
  let stageHeatmap = document.getElementById('stageHeatmap')
  let stageAnnotated = document.getElementById('stageAnnotated')

  stageOriginal.src = base + (result.original || 'original.jpg')
  stageSegmented.src = base + (result.car_segmented || 'car_segmented.jpg')
  stageHeatmap.src = base + (result.heatmap || 'heatmap.jpg')
  stageAnnotated.src = base + (result.annotated || 'annotated.jpg')

  pipelineStages.style.display = ''

  // Set annotated as default main view
  pic.src = base + (result.annotated || 'annotated.jpg')
  pic.onload = () => {
    overlay.width = pic.naturalWidth
    overlay.height = pic.naturalHeight
  }

  // Mark annotated as active
  document.querySelectorAll('.stage-card').forEach(c => c.classList.remove('active'))
  document.querySelector('.stage-card[data-stage="annotated"]').classList.add('active')

  // Click handlers for stage cards
  document.querySelectorAll('.stage-card').forEach(card => {
    card.onclick = () => {
      let img = card.querySelector('img')
      if (!img.src) return
      pic.src = img.src
      pic.onload = () => {
        overlay.width = pic.naturalWidth
        overlay.height = pic.naturalHeight
        // Redraw boxes only if viewing annotated
        if (card.dataset.stage === 'annotated') {
          drawBoxes()
        } else {
          let ctx = overlay.getContext('2d')
          ctx.clearRect(0, 0, overlay.width, overlay.height)
        }
      }
      document.querySelectorAll('.stage-card').forEach(c => c.classList.remove('active'))
      card.classList.add('active')
    }
  })
}

function mapDefects(rawDefects, cropOffset) {
  let ox = cropOffset[0] || 0
  let oy = cropOffset[1] || 0

  return rawDefects.map((d, i) => {
    let confidence = d.confidence || 0
    let severity
    if (confidence > 0.8) severity = 'критический'
    else if (confidence > 0.5) severity = 'средний'
    else severity = 'мелкий'

    let typeMap = {
      'scratch': 'царапина',
      'spot': 'дефект',
      'dent': 'вмятина',
      'chip': 'дефект'
    }

    return {
      id: i + 1,
      type: typeMap[d.type] || d.label || d.type,
      zone: 'кузов',
      severity: severity,
      x: (d.bbox ? d.bbox[0] : 0) + ox,
      y: (d.bbox ? d.bbox[1] : 0) + oy,
      w: d.bbox ? d.bbox[2] : 0,
      h: d.bbox ? d.bbox[3] : 0,
      confidence: confidence
    }
  })
}

/* ---------- Drawing ---------- */

function drawBoxes() {
  let ctx = overlay.getContext('2d')
  ctx.clearRect(0, 0, overlay.width, overlay.height)

  defects.forEach(d => {
    let color = SEV_COLORS[d.severity]
    ctx.strokeStyle = color
    ctx.lineWidth = 2
    ctx.strokeRect(d.x, d.y, d.w, d.h)

    ctx.fillStyle = color
    let label = `#${d.id} ${d.type}`
    ctx.font = '600 13px JetBrains Mono, monospace'
    let tw = ctx.measureText(label).width
    ctx.fillRect(d.x, d.y - 20, tw + 10, 20)

    ctx.fillStyle = '#0c0b0a'
    ctx.fillText(label, d.x + 5, d.y - 6)
  })
}

function showResults() {
  if (!defects.length) {
    results.style.display = 'block'
    defectList.innerHTML = '<div class="no-defects">дефектов не обнаружено</div>'
    return
  }

  results.style.display = 'block'
  exportBtn.style.display = ''
  defectList.innerHTML = ''

  defects.forEach(d => {
    let row = document.createElement('div')
    row.className = 'defect-row'

    row.innerHTML = `
      <span class="defect-id">#${d.id}</span>
      <span class="sev-dot" style="background:${SEV_COLORS[d.severity]}"></span>
      <span class="defect-type">${d.type}</span>
      <span class="defect-zone">${d.zone}</span>
      <span class="defect-conf">${Math.round(d.confidence * 100)}%</span>
    `

    row.onmouseenter = () => highlightBox(d, true)
    row.onmouseleave = () => highlightBox(d, false)

    defectList.appendChild(row)
  })
}

function highlightBox(d, on) {
  let ctx = overlay.getContext('2d')
  drawBoxes()
  if (on) {
    ctx.strokeStyle = '#fff'
    ctx.lineWidth = 3
    ctx.strokeRect(d.x - 2, d.y - 2, d.w + 4, d.h + 4)
  }
}

/* ---------- Export ---------- */

exportBtn.onclick = () => {
  if (!cur || !defects.length) return notify('нечего экспортировать')

  let im = new Image()
  im.crossOrigin = 'anonymous'
  im.onload = () => {
    let c = document.createElement('canvas')
    c.width = im.naturalWidth
    c.height = im.naturalHeight
    let ctx = c.getContext('2d')

    ctx.drawImage(im, 0, 0, c.width, c.height)
    ctx.drawImage(overlay, 0, 0, c.width, c.height)

    let a = document.createElement('a')
    a.download = 'comp2_report_' + cur + '.jpg'
    a.href = c.toDataURL('image/jpeg', 0.92)
    a.click()
    notify('отчёт сохранён')
  }
  im.src = pic.src
}

/* ---------- Rotate ---------- */

let rotDeg = 0
let rotating = false

document.getElementById('rotateBtn').onclick = async () => {
  if (!cur || rotating) return
  rotating = true

  // Animate rotation visually
  rotDeg += 90
  pic.style.transition = 'transform .35s ease'
  pic.style.transform = 'rotate(' + rotDeg + 'deg)'

  try {
    let resp = await fetch('/api/photos/' + cur + '/rotate', { method: 'POST' })
    if (!resp.ok) throw new Error('Ошибка поворота')

    // Wait for animation to finish, then swap to real rotated image
    await new Promise(r => setTimeout(r, 400))

    pic.style.transition = 'none'
    pic.style.transform = 'none'
    rotDeg = 0

    let ts = Date.now()
    pic.src = curSrc + '?t=' + ts
    document.querySelectorAll('.th img').forEach(img => {
      if (img.src.includes(cur)) img.src = curSrc + '?t=' + ts
    })
    clearResults()
    notify('повёрнуто на 90°')
  } catch (e) {
    pic.style.transition = 'none'
    pic.style.transform = 'none'
    rotDeg = 0
    notify('Ошибка: ' + e.message)
  }
  rotating = false
}

/* ---------- Sidebar ---------- */

document.getElementById('all').onclick = () => {
  if (sel.size === allPhotos.length) sel.clear()
  else allPhotos.forEach(x => sel.add(x.id))
  renderGrid()
}

document.getElementById('del').onclick = async () => {
  if (!sel.size) return notify('сначала выберите')

  for (let id of sel) {
    try { await fetch('/api/photos/' + id, { method: 'DELETE' }) } catch {}
  }

  if (sel.has(cur)) {
    cur = null
    curSrc = null
    cwrap.style.display = 'none'
    emp.style.display = ''
    bbar.style.display = 'none'
    results.style.display = 'none'
    pipelineStages.style.display = 'none'
  }
  sel.clear()
  render()
}

render()

/* --- Sidebar toggle --- */
const sidebarToggle = document.getElementById('sidebarToggle')
const sidebar = document.getElementById('sidebar')
const panelEl = document.querySelector('.panel')

sidebarToggle.addEventListener('click', () => {
  sidebar.classList.toggle('collapsed')
  panelEl.classList.toggle('sidebar-collapsed')
})
