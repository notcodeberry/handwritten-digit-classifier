import { API_URL } from "./config.js";

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const clearBtn = document.getElementById("clearBtn");
const predictBtn = document.getElementById("predictBtn");
const brushRange = document.getElementById("brushRange");
const resultP = document.getElementById("result");

function initCanvas() {
	ctx.fillStyle = "black";
	ctx.fillRect(0, 0, canvas.width, canvas.height);
	ctx.lineCap = "round";
	ctx.lineJoin = "round";
	ctx.strokeStyle = "white";
	ctx.lineWidth = Number(brushRange.value);
}

initCanvas();

let drawing = false;
let lastX = 0,
	lastY = 0;

function getPointerPos(evt) {
	const rect = canvas.getBoundingClientRect();
	if (evt.touches && evt.touches.length > 0) {
		return {
			x: evt.touches[0].clientX - rect.left,
			y: evt.touches[0].clientY - rect.top,
		};
	} else {
		return {
			x: evt.clientX - rect.left,
			y: evt.clientY - rect.top,
		};
	}
}

function startDraw(evt) {
	evt.preventDefault();
	drawing = true;
	const pos = getPointerPos(evt);
	lastX = pos.x;
	lastY = pos.y;
}

function draw(evt) {
	if (!drawing) return;
	evt.preventDefault();
	const pos = getPointerPos(evt);
	ctx.beginPath();
	ctx.moveTo(lastX, lastY);
	ctx.lineTo(pos.x, pos.y);
	ctx.stroke();
	lastX = pos.x;
	lastY = pos.y;
}

function endDraw(evt) {
	evt.preventDefault();
	drawing = false;
}

// mouse events
canvas.addEventListener("mousedown", startDraw);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", endDraw);
canvas.addEventListener("mouseleave", endDraw);

// touch events
canvas.addEventListener("touchstart", startDraw, { passive: false });
canvas.addEventListener("touchmove", draw, { passive: false });
canvas.addEventListener("touchend", endDraw, { passive: false });
canvas.addEventListener("touchcancel", endDraw, { passive: false });

brushRange.addEventListener("input", () => {
	ctx.lineWidth = Number(brushRange.value);
});

clearBtn.addEventListener("click", () => {
	initCanvas();
	resultP.textContent = "";
});

predictBtn.addEventListener("click", async () => {
	const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data;

	let sum = 0;

	for (let i = 0; i < pixels.length; i += 4) sum += pixels[i]; // r chanel (black background -> 0)

	if (sum < 2500) {
		resultP.textContent = "There is nothing to detect here.";
		return;
	}

	const dataURL = canvas.toDataURL("image/png");

	resultP.textContent = "Sending to a model...";

	try {
		const resp = await fetch(API_URL, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ encodedImage: dataURL }),
		});

		if (!resp.ok) throw new Error("Server error");

		const json = await resp.json();
		// Expected response: { prediction: <0-9|null>, confidence: 0.xx, message?: "..." }
		if (json.prediction === null || json.prediction === undefined) {
			resultP.textContent = json.message || "Digit not recognized";
		} else {
			const conf = json.confidence
				? ` (accuracy ${(json.confidence * 100).toFixed(1)}%)`
				: "";
			resultP.textContent = `This looks like a digit: ${json.prediction}${conf}`;
		}
	} catch (err) {
		console.log(err);
		resultP.textContent = "Server communication error, see the console.";
	}
});
