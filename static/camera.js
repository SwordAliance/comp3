const vid = document.getElementById('vid')
const cvs = document.getElementById('cvs')
const fl = document.getElementById('fl')
const hint = document.getElementById('hint')
const cnt = document.getElementById('cnt')
const ntf = document.getElementById('ntf')

let stream = null
let facing = 'environment'

function notify(t) {
  ntf.textContent = t
  ntf.classList.add('pop')
  setTimeout(() => ntf.classList.remove('pop'), 1800)
}

async function updCount() {
  try {
    let res = await fetch('/api/photos')
    let arr = await res.json()
    cnt.textContent = arr.length
  } catch { cnt.textContent = '?' }
}

async function upload(dataUrl) {
  try {
    await fetch('/api/photos', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ data: dataUrl })
    })
    return true
  } catch { return false }
}

async function go() {
  if (stream) stream.getTracks().forEach(t => t.stop())

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: facing, width: { ideal: 1920 }, height: { ideal: 1080 } },
      audio: false
    })
    vid.srcObject = stream
    hint.classList.add('off')
  } catch(e) {
    console.error(e)
    notify('камера недоступна')
  }
}

async function snap() {
  if (!stream) return

  let track = stream.getVideoTracks()[0]
  let s = track.getSettings()
  cvs.width = s.width || vid.videoWidth
  cvs.height = s.height || vid.videoHeight

  let ctx = cvs.getContext('2d')
  ctx.drawImage(vid, 0, 0, cvs.width, cvs.height)
  let url = cvs.toDataURL('image/jpeg', 0.85)

  fl.classList.add('go')
  fl.onanimationend = () => fl.classList.remove('go')

  let ok = await upload(url)
  if (ok) {
    updCount()
    notify('сохранено')
  } else {
    notify('ошибка сохранения')
  }
}

hint.onclick = go
document.getElementById('snap').onclick = snap
document.getElementById('flip').onclick = () => {
  facing = facing === 'environment' ? 'user' : 'environment'
  go()
}

const fileIn = document.getElementById('fileIn')

document.getElementById('gallery').onclick = () => fileIn.click()

fileIn.onchange = async () => {
  let files = Array.from(fileIn.files)
  if (!files.length) return

  let ok = 0
  for (let file of files) {
    let dataUrl = await new Promise(resolve => {
      let r = new FileReader()
      r.onload = () => resolve(r.result)
      r.readAsDataURL(file)
    })
    if (await upload(dataUrl)) ok++
  }

  updCount()
  notify(ok === 1 ? 'фото загружено' : `загружено ${ok}`)
  fileIn.value = ''
}

updCount()
go()
