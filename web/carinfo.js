
/**
 * Module dependencies.
 */

var express = require('express')
	, routes = require('./routes')
	, mysql = require('mysql')
	, async = require('async')
	, path = require('path')
	, partials = require('express-partials')

var app = module.exports = express.createServer();

// Configuration

app.configure(function(){
	app.set('views', path.join(__dirname, 'views'));
	app.set('view engine', 'ejs');
	app.use(express.bodyParser());
	app.use(express.methodOverride());
	app.use(app.router);
	app.use(partials());
	app.use(express.static(path.join(__dirname, 'public')));
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

client.query('use car', function(err, res, fields) {
	if (err) { throw err; }
});


// Routes 
app.get('/', routes.index);

app.get('/spec/:spec/', function(req, res){
	console.log(req.params.spec);
	spec = req.params.spec;
	graph = {};
	graph["spec"] = {
		"name":spec,
		"hot":54,
		"sentiments": []
	};
	nodes = [];
	keywords = [];
	sentiments = []; // good and neutral and bad!
	result_comment = [];
	result_keyword = [];
	result_sentiment = [];


	node_list = ["space","power","operation","oilwear","comfort","appearance","decoration","costperformance","failure","maintenance"];
	tran_tag = {'space': '空间', 'power': '动力', 'operation': '操控', 'oilwear': '油耗', 'comfort': '舒适性', 'appearance': '外观', 'decoration': '内饰', 'costperformance': '性价比','failure': '故障', 'maintenance': '保养'};
	rev_tran_tag = {};
	tag_idx = {};

	for (i in node_list) {
		tag_idx[node_list[i]] = i;
	};
	for (i in tran_tag) {
		rev_tran_tag[tran_tag[i]] = i;
	};
	// tag_idx = {'space': 0, 'power': 1, 'operation': 2, 'oilwear': 3, 'comfort': 4, 'appearance': 5, 'decoration': 6, 'costperformance': 7,'failure': 8, 'maintenance': 9};
	tran_web = {'autohome': '汽车之家', 'yiche': '易车网', 'pcauto': '太平洋汽车', 'xgo': '汽车点评网', 'sohu': '搜狐汽车', 'netease': '网易汽车', 'xcar': '爱卡汽车'}
	// Note about this for async loop.
	// 1. use async.each / series.
	// 2. use promise.
	//    different interface in jquery(not /A+), standard, when.js, then.js, Q, co.
	//    .then() receive either async / sync. Remember resolve / reject.
	// 3. use co. iterator and generator.

	var idx = 0, cnt = 0;
	node_item = [];
	for (i in node_list)
		node_item.push({"name": tran_tag[node_list[i]], "weight": 0, "comments": []});


	function get_comments() { 
		return new Promise(function(resolve, reject) {
			sql = 'select distinct * from show_comment where spec="' + spec + '"and first_show != 0 order by first_show desc;'
			client.query(sql, function(err, data) {
				result_comment = data;
				resolve();
			});
		});
	};


	function get_keywords() {
		return new Promise(function(resolve, reject) {
			sql = 'select keyword, weight from show_keyword where spec="' + spec + '" order by weight desc;'
			client.query(sql, function(err, data) {
				result_keyword = data;
				resolve();
			});
		});
	};

	function get_sentiments() { 
		return new Promise(function(resolve, reject) {
			sql = 'select * from show_sentiments where spec="' + spec + '";'
			client.query(sql, function(err, data) {
				result_sentiment = data;
				resolve();
			});
		});
	};

	function push_data() {
		return new Promise(function(resolve, reject) {
			result_comment.forEach(function(data){
				comment = {"date": data.date, "comment": data.comment, "web": tran_web[data.web], "url": data.url};
				node_item[tag_idx[data.label]].comments.push(comment);
			});
			for (i in node_item) {
				if (node_item[i].comments.length != 0) {
					nodes.push(node_item[i]);
				}
			};
			result_keyword.forEach(function(data){
				keywords.push({"weight": data.weight, "keyword":data.keyword});
			});

			total_good = 0
			total_neutral = 0
			total_bad = 0
			node_list.forEach(function(i){
				data = result_sentiment[0][i]
				good = parseInt(data.split("/")[0]);
				neutral = parseInt(data.split("/")[1]);
				bad = parseInt(data.split("/")[2]);
				total_good += good
				total_neutral += neutral
				total_bad += bad
				sentiments.push([good+neutral+bad, good, neutral, bad]);
			})
			console.log("?");

			graph["nodes"] = nodes;
			graph["keywords"] = keywords;
			graph["sentiments"] = sentiments;
			graph["spec"]["sentiments"] = [total_good+total_neutral+total_bad, total_good, total_neutral, total_bad];
			console.log(graph);
			res.json(graph);
			
		});
	}; 


	get_comments().then(get_keywords).then(get_sentiments).then(push_data);

	// new Promise(get_comments).then(function() {
	// 	return new Promise(store_comments);
	// });

});

app.get('/spec/:spec/load_more/:label/:offset', function(req, res) {
	spec = req.params.spec;
	offset = req.params.offset;
	label = req.params.label;
	label = rev_tran_tag[label];
	sql = "select * from car.show_comment where first_show = 0 and spec = '" + spec + "' and label = '" + label + "' limit " + offset + " , 2"
	console.log(sql);
	client.query(sql, function(err, result){
		result.forEach(function(data) {
			data.web = tran_web[data.web];
		});
		res.json(result);
	}); 
});

app.listen(8000, function(){
	console.log("Express server listening on port %d in %s mode", app.address().port, app.settings.env);
});
