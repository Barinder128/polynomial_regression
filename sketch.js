let x_axis = [];    // x axis of points marked by mouse click on canvas
let y_axis = [];    // y axis of points marked by mouse click on canvas

let a_ts;
let b_ts;
let c_ts;     
let d_ts;           // y-intercept tensor
let dragging = false;

const learningRate = 0.2;
const optimizer = tf.train.adam(learningRate);   // Stochastic Gradient Descent
                                                // is the optimizer with a learning rate


function setup() {
  createCanvas(600, 600);
  a_ts = tf.variable(tf.scalar(random(-1, 1)));
  b_ts = tf.variable(tf.scalar(random(-1, 1)));
  c_ts = tf.variable(tf.scalar(random(-1, 1)));
  d_ts = tf.variable(tf.scalar(random(-1, 1)));
}

function loss(pred, labels) {
  return pred
    .sub(labels)
    .square()
    .mean();
}



function predict(x){
  const x_tsr = tf.tensor1d(x);
  const y_pred_tsr = x_tsr.pow(tf.scalar(3)).mul(a_ts).add(x_tsr.square().mul(b_ts)).add(x_tsr.mul(c_ts)).add(d_ts) ;   // y=ax^3+bx^2+cx+d
  return y_pred_tsr;
}

function mousePressed() {
  dragging = true;
}

function mouseReleased(){
  dragging = false;
}

function draw() {
  if(dragging){
    let x = map(mouseX, 0, width, -1, 1);    //x and y data-points are normalized
    let y = map(mouseY, 0, height, 1, -1);
    x_axis.push(x);
    y_axis.push(y);
  }
  else{
    // code for training model
    if(x_axis.length>0){
      tf.tidy(() => {
          const y_train_tsr = tf.tensor1d(y_axis);
          optimizer.minimize(() => loss(predict(x_axis), y_train_tsr));
       });
    }
  }
    
  background(0);
  // code to draw data points
  stroke(255);
  strokeWeight(8);
  for(let i =0; i<x_axis.length;i++){
    let px = map(x_axis[i], -1, 1, 0, width);
    let py = map(y_axis[i], -1, 1, height, 0);
    point(px, py);
  }
  // code for prediction
  line_data_x = [];
  for(i = -1; i<1.02; i+=0.02){
    line_data_x.push(i);
  }

  const y_pred_tsr = tf.tidy(() => predict(line_data_x));
  let line_data_y = y_pred_tsr.dataSync();
  y_pred_tsr.dispose();
  noFill();
  strokeWeight(3);
  beginShape();

  for(let i=0; i<line_data_x.length; i++){
    vertex(map(line_data_x[i], -1, 1, 0, width), map(line_data_y[i], -1, 1, height, 0))
  }
  endShape();

   
}