images = ['https://media.istockphoto.com/photos/baby-sheep-close-up-picture-id1164046558?s=170667a',
  'https://www.pngitem.com/pimgs/m/189-1892631_cat-head-pixel-art-hd-png-download.png', 'https://daily.jstor.org/wp-content/uploads/2020/09/corn_is_everywhere_1050x700.jpg',
    'https://static.wikia.nocookie.net/terraria-tremor/images/5/5d/EvilCorn.png/revision/latest/top-crop/width/360/height/450?cb=20170306122056']


var data = [{
        x: [1, 3, 3, 4],
        y: [1, 2, 4, 4],
        z: [1, 2, 3, 5],
        mode: 'markers',
        type: 'scatter3d',
        marker: {
          color: 'rgb(23, 190, 207)',
          size: 2
        }
    }];

Plotly.newPlot('graphDiv', data, {
  hovermode: 'closest'
});

document.getElementById('graphDiv').on('plotly_hover', function(data) {
  document.getElementById('image').innerHTML="<img src='" + images[data.points[0].pointNumber] + "' />";
});