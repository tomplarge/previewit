var song = '../audio/03 All My Loving copy.m4a';
var startTimes = [0, 25, 49.5, 74.5, 99, 126]; // last element must be end time of song

// create gradient for canvas
var ctx = document.createElement('canvas').getContext('2d');
var linGrad = ctx.createLinearGradient(0, 56, 0, 200); // why does 56 center it?
linGrad.addColorStop(0.5, 'rgba(255, 255, 255, 1.000)');
linGrad.addColorStop(0.5, 'rgba(183, 183, 183, 1.000)');

// create waveform object and load audio
var wavesurfer = WaveSurfer.create({
    container: '#waveform',
    waveColor: linGrad,
    progressColor: 'rgba(100, 100, 100, 1.000)',
    barWidth: 3
});
wavesurfer.load(song);

// add colored region for each section
function addRegions() {
	var hue = 0;
	for (i = 0; i < startTimes.length - 1; i++) {
		wavesurfer.addRegion({
    		start: startTimes[i],
    		end: startTimes[i+1],
    		color: 'hsla(' + hue + ', 80%, 30%, .4)',
    		drag: false,
    		resize: false
  		});
  		hue = (hue + 60) % 360
	}

	var regions = document.getElementsByClassName('wavesurfer-region');
	for (i = 0; i < regions.length; i++) {
		regions[i].style.zIndex = 0;
	}
}

wavesurfer.on('ready', function () {
    wavesurfer.play(); //autoplay
    addRegions();
});

// button listeners
document.getElementById('play').addEventListener('click', function(){
    wavesurfer.play();
});

document.getElementById('pause').addEventListener('click', function(){
    wavesurfer.pause();
});

document.getElementById('skip').addEventListener('click', function(){
	var currTime = wavesurfer.getCurrentTime();
	for (i = 0; i < startTimes.length; i++) {
		if (startTimes[i] > currTime) {
			wavesurfer.play(startTimes[i]);
			break;
		}
	}
});

