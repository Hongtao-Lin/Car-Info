
/**
 * Module dependencies.
 */

var express = require('express')
  , routes = require('./routes')
  , mysql = require('mysql')
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
  password: '1234'
});
client.connect();

client.query('use mysql', function(err, res, fields) {
  if (err) { throw err; }
  // console.log(res);
});

// graph = {};
//   graph["series"] = {
//     "id":0,
//     "name":req.body,
//     "hot":54,
//     "good":82,
//     "bad":7,
//     "neutual":11,
//     "group":0
//   };
//   graph["nodes"] = [];
//   node_list = ["space","power","operation","oilwear","comfort","appearance","decoration","costperformance","failure","maintance"];
//   for (i in node_list) {
//     client.query('select * from info where series="' + req.body.series + '" and label="' + node_list[i] + '"', function(err, res, fields) {
//       if (err) { 
//         // return to 500 page.
//         throw err; 
//       }
      
//       comments = [];
//       for (j in result) {
//         comments.push({"date":result[j].date,"content":result[j].comment,"from":result[j].from,"url":result[j].url});
//       };
//       node = {"id":int(i)+1,"name":node_list[i],"group":1,"weight":0,"comments":comments};
//       console.log(node);
//     });
//   }; 
// Routes

app.get('/', routes.index);

// app.get('/', function(req, res){
//   console.log(1);
//   res.body = {"series":"testspec"};
//   console.log(req.body);
//   graph = {};
//   graph["series"] = {
//     "id":0,
//     "name":req.body,
//     "hot":54,
//     "good":82,
//     "bad":7,
//     "neutual":11,
//     "group":0
//   };
//   nodes = [];
//   node_list = ["space","power","operation","oilwear","comfort","appearance","decoration","costperformance","failure","maintance"];
//   for (i in node_list) {
//     client.query('select * from info where series="' + req.body.series + '" and label="' + node_list[i] + '"', function(err, res, fields) {
//       if (err) { 
//         // return to 500 page.
//         throw err; 
//       }
//       console.log(res);
      
//       comments = [];
//       for (i in res) {
//         comments.push({"date":res[i].date,"content":res[i].comment,"from":res[i].from,"url":res[i].url});
//       };
//       node = {"id":i+1,"name":node_list[i],"group":1,"weight":0,"comments":comments};
//       nodes.push(node);
//     });
//   }; 
//   graph["nodes"] = nodes;
//   // then we can access mysql to transform the value into json data.
//   res.render('visual', {
//     layout: '',
//     graph: graph
//   });
// });

app.listen(8000, function(){
  console.log("Express server listening on port %d in %s mode", app.address().port, app.settings.env);
});
