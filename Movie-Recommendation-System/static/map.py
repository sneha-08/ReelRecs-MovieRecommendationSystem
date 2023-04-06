// Define a list of relevant movies for a user
var relevant_movies = ["The Dark Knight", "Inception", "Interstellar"];

// Get a list of recommended movies for the user
var recommended_movies = ["The Dark Knight Rises", "The Prestige", "Dunkirk"];

// Calculate the precision and recall for each recommended movie
var precision_recall = [];
var tp = 0;
var fp = 0;
for (var i = 0; i < recommended_movies.length; i++) {
  var movie = recommended_movies[i];
  if (relevant_movies.includes(movie)) {
    tp += 1;
  } else {
    fp += 1;
  }
  var precision = tp / (tp + fp);
  var recall = tp / relevant_movies.length;
  precision_recall.push([precision, recall]);
}
  
// Sort the precision-recall pairs by decreasing recall
precision_recall.sort(function(a, b) {
  return b[1] - a[1];
});

// Calculate the area under the precision-recall curve
var auc = 0;
var prev_recall = 0;
for (var i = 0; i < precision_recall.length; i++) {
  var precision = precision_recall[i][0];
  var recall = precision_recall[i][1];
  auc += (recall - prev_recall) * precision;
  prev_recall = recall;
}

// Plot the precision-recall curve
var trace = {
  x: precision_recall.map(function(x) { return x[1]; }),
  y: precision_recall.map(function(x) { return x[0]; }),
  type: 'scatter',
  mode: 'lines',
  line: {
    color: 'blue'
  },
  fill: 'tozeroy',
  name: 'AUC: ' + auc.toFixed(2)
};
var data = [trace];
var layout = {
  title: 'Precision-Recall Curve',
  xaxis: {
    title: 'Recall'
  },
  yaxis: {
    title: 'Precision'
  }
};
Plotly.newPlot('plot', data, layout);
