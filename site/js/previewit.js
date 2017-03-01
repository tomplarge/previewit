var song = '../audio/03 All My Loving copy.m4a';
// first element must be zero, last element must be end time of song
var startTimes = [0, 25, 49.5, 61.5, 74.5, 99, 129.5];
var sections = [0, 0, 1, 2, 0, 1];
var numSections = 3;

// create gradient for canvas
var ctx = document.createElement('canvas').getContext('2d');
var linGrad = ctx.createLinearGradient(0, 56, 0, 200); // why does 56 center it?
linGrad.addColorStop(0.5, 'rgba(255, 255, 255, 1.000)');
linGrad.addColorStop(0.5, 'rgba(183, 183, 183, 1.000)');

// create waveform object and load audio
var wavesurfer = WaveSurfer.create({
    container: '#waveform',
    waveColor: linGrad,
    progressColor: 'rgba(64,64,64,1)',//'rgba(2, 56, 88, 1.000)', //
    barWidth: 3
});
wavesurfer.load(song);

// add colored region for each section
/*
var hues = [];
for (i = 0; i < numSections; i++) {
	hues.push(i*360/numSections); // hue is out of 360
}
*/
//hues = ['#8dd3c7', '#bebada', '#fb8072', '#80b1d3', '#fdb462', ,'#ffffb3'];
hues = ['rgba(4,90,141,0.7)', 'rgba(116,169,207,0.7)', 'rgba(2,56,88,.7)', 'rgba(54,144,192,0.7)', 'rgba(116,169,207,0.7)', 'rgba(166,189,219,0.7)', 'rgba(5,112,176,0.7)'];
function addRegions() {
	wavesurfer.clearRegions();
	for (i = 0; i < startTimes.length - 1; i++) {
		wavesurfer.addRegion({
    		start: startTimes[i],
    		end: startTimes[i+1],
    		//color: 'hsla(' + hues[sections[i]] + ', 80%, 30%, .4)',
    		color: hues[sections[i]],
    		drag: false,
    		resize: false
  		});
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
var playState = true;
document.getElementById('play-pause').addEventListener('click', function(){
	if (playState) {
		wavesurfer.pause()
		document.getElementById('play-pause').innerHTML = '<i class="material-icons">play_arrow</i>';
		playState = false;
	}
	else {
		wavesurfer.play();
		document.getElementById('play-pause').innerHTML = '<i class="material-icons">pause</i>';
		playState = true;
	}
    
});

document.getElementById('skip').addEventListener('click', function(){
	var currTime = wavesurfer.getCurrentTime();
	for (i = 0; i < startTimes.length; i++) {
		if (startTimes[i] > currTime) {
			wavesurfer.skip(startTimes[i] - currTime);
			break;
		}
	}
});

document.getElementById('back').addEventListener('click', function(){
	var currTime = wavesurfer.getCurrentTime();
	for (i = 0; i < startTimes.length; i++) {
		if (startTimes[i] >= currTime) {
			wavesurfer.skip(startTimes[i-1] - currTime);
			break;
		}
	}
});

// alignment and resizing
function alignTrackList() {
	var $container = $('div.album-info');
	var $top = $('div.center-align');
	var $bot = $('div.left-align');
	var containerHeight = $container.height();
	var topHeight = $top.height();
	$bot.css('height', + (containerHeight-topHeight) + "px");
}

alignTrackList();
$(window).resize(function(){
	wavesurfer.empty();
	wavesurfer.drawBuffer();
	addRegions();
	playState = false; // resizing pauses player
	document.getElementById('play-pause').innerHTML = '<i class="material-icons">play_arrow</i>';
	alignTrackList();
});


