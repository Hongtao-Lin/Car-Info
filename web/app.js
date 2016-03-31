
/**
 * Module dependencies.
 */

var express = require('express')
  , routes = require('./routes')
  , mysql = require('mysql')
  , async = require('async')
  , partials = require('express-partials')

var app = module.exports = express.createServer();

// Configuration

app.configure(function(){
  app.set('views', __dirname + '/views');
  app.set('view engine', 'ejs');
  app.use(express.bodyParser());
  app.use(express.methodOverride());
  app.use(app.router);
  app.use(partials());
  app.use(express.static(__dirname + '/public'));
});

app.configure('development', function(){
  app.use(express.errorHandler({ dumpExceptions: true, showStack: true }));
});

app.configure('production', function(){
  app.use(express.errorHandler());
});

// get mysql running!
var client = mysql.createConnection({
  user: 'root',
  password: ''
});
client.connect();

client.query('use car', function(err, res, fields) {
  if (err) { throw err; }
  // console.log(res);
});


// Routes 
app.get('/', routes.index);

app.get('/spec/:spec/', function(req, res){
  console.log(req.params.spec);
  spec = req.params.spec;
  graph = {};
  graph["series"] = {
    "id":0,
    "name":spec,
    "hot":54,
    "good":82,
    "bad":7,
    "neutual":11,
    "group":0
  };
  nodes = [];
  node_list = ["space","power","operation","oilwear","comfort","appearance","decoration","costperformance","failure","maintance"];
  tran_tag={'space': '空间', 'power': '动力', 'operation': '操控', 'oilwear': '油耗', 'comfort': '舒适性', 'appearance': '外观', 'decoration': '内饰', 'costperformance': '性价比','failure': '故障', 'maintance': '保养'};
  var i = 0;

  // Note about this for async loop.
  // 1. use async.each / series.
  // 2. use promise.
  //    different interface in jquery(not /A+), standard, when.js, then.js, Q, co.
  //    .then() receive either async / sync. Remember resolve / reject.
  // 3. use co. iterator and generator.
  // 4. use 

  function step1(resolve, reject) { 
    async.eachSeries(node_list, 
    function(node, callback){
      client.query('select * from info where spec="' + spec + '" and label="' + node + '" limit 5;', function(err, data) {
        i += 1;
        if (err) { 
          callback(err);
        } 
        else if(data.length == 0) {
          callback();
        }
        else {
          comments = [];
          for (j in data) {
            if (data[j].comment != '')
	      comments.push({"date":data[j].date,"content":data[j].comment,"from":data[j].web,"url":data[j].url});
          };
          node_item = {"id":i,"name":tran_tag[node_list[i-1]],"group":1,"weight":0,"comments":comments};
          nodes.push(node_item);
          callback();
        }
      });
    },
    function(err) {
      if (err) {
        throw err;
      }
      else {
        graph["nodes"] = nodes;
        resolve();
      }
    }
    );
  };

  function step2(resolve, reject) {
    console.log(graph);
    res.json(graph);
  }; 

  new Promise(step1).then(function() {
    return new Promise(step2);
  });

});

app.listen(8000, function(){
  console.log("Express server listening on port %d in %s mode", app.address().port, app.settings.env);
});
