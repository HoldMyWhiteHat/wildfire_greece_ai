fetch("/static/maps.json")
  .then(res => res.json())
  .then(data => {
    window.maps = data;
    loadMap(0);
  });

function loadMap(index) {
  const iframe = document.getElementById("map-frame");
  iframe.src = "/static/visualization/maps/" + maps[index].file;
}

    slider.oninput = e => load(e.target.value);

    document.getElementById("next").onclick = () => {
      slider.value = Math.min(+slider.value + 1, slider.max);
      load(slider.value);
    };

    document.getElementById("prev").onclick = () => {
      slider.value = Math.max(+slider.value - 1, slider.min);
      load(slider.value);
    };

   // loadMap(0);
