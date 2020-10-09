(function() {
  var fn = function() {
    
    (function(root) {
      function now() {
        return new Date();
      }
    
      var force = false;
    
      if (typeof root._bokeh_onload_callbacks === "undefined" || force === true) {
        root._bokeh_onload_callbacks = [];
        root._bokeh_is_loading = undefined;
      }
    
      
      
    
      var element = document.getElementById("c1248422-ac16-4d25-90ac-4d7a79302184");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid 'c1248422-ac16-4d25-90ac-4d7a79302184' but no matching script tag was found.")
        }
      
    
      function run_callbacks() {
        try {
          root._bokeh_onload_callbacks.forEach(function(callback) {
            if (callback != null)
              callback();
          });
        } finally {
          delete root._bokeh_onload_callbacks
        }
        console.debug("Bokeh: all callbacks have finished");
      }
    
      function load_libs(css_urls, js_urls, callback) {
        if (css_urls == null) css_urls = [];
        if (js_urls == null) js_urls = [];
    
        root._bokeh_onload_callbacks.push(callback);
        if (root._bokeh_is_loading > 0) {
          console.debug("Bokeh: BokehJS is being loaded, scheduling callback at", now());
          return null;
        }
        if (js_urls == null || js_urls.length === 0) {
          run_callbacks();
          return null;
        }
        console.debug("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
        root._bokeh_is_loading = css_urls.length + js_urls.length;
    
        function on_load() {
          root._bokeh_is_loading--;
          if (root._bokeh_is_loading === 0) {
            console.debug("Bokeh: all BokehJS libraries/stylesheets loaded");
            run_callbacks()
          }
        }
    
        function on_error() {
          console.error("failed to load " + url);
        }
    
        for (var i = 0; i < css_urls.length; i++) {
          var url = css_urls[i];
          const element = document.createElement("link");
          element.onload = on_load;
          element.onerror = on_error;
          element.rel = "stylesheet";
          element.type = "text/css";
          element.href = url;
          console.debug("Bokeh: injecting link tag for BokehJS stylesheet: ", url);
          document.body.appendChild(element);
        }
    
        const hashes = {"https://cdn.bokeh.org/bokeh/release/bokeh-2.2.1.min.js": "qkRvDQVAIfzsJo40iRBbxt6sttt0hv4lh74DG7OK4MCHv4C5oohXYoHUM5W11uqS", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.1.min.js": "Sb7Mr06a9TNlet/GEBeKaf5xH3eb6AlCzwjtU82wNPyDrnfoiVl26qnvlKjmcAd+", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.1.min.js": "HaJ15vgfmcfRtB4c4YBOI4f1MUujukqInOWVqZJZZGK7Q+ivud0OKGSTn/Vm2iso"};
    
        for (var i = 0; i < js_urls.length; i++) {
          var url = js_urls[i];
          var element = document.createElement('script');
          element.onload = on_load;
          element.onerror = on_error;
          element.async = false;
          element.src = url;
          if (url in hashes) {
            element.crossOrigin = "anonymous";
            element.integrity = "sha384-" + hashes[url];
          }
          console.debug("Bokeh: injecting script tag for BokehJS library: ", url);
          document.head.appendChild(element);
        }
      };
    
      function inject_raw_css(css) {
        const element = document.createElement("style");
        element.appendChild(document.createTextNode(css));
        document.body.appendChild(element);
      }
    
      
      var js_urls = ["https://cdn.bokeh.org/bokeh/release/bokeh-2.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.1.min.js"];
      var css_urls = [];
      
    
      var inline_js = [
        function(Bokeh) {
          Bokeh.set_log_level("info");
        },
        
        function(Bokeh) {
          (function() {
            var fn = function() {
              Bokeh.safely(function() {
                (function(root) {
                  function embed_document(root) {
                    
                  var docs_json = '{"f598b355-bd45-43d1-8c4b-292c852d4102":{"roots":{"references":[{"attributes":{"children":[{"id":"4276"},{"id":"4274"}]},"id":"4277","type":"Column"},{"attributes":{"axis_label":"ESS","formatter":{"id":"4259"},"ticker":{"id":"4210"}},"id":"4209","type":"LinearAxis"},{"attributes":{"toolbar":{"id":"4275"},"toolbar_location":"above"},"id":"4276","type":"ToolbarBox"},{"attributes":{"callback":null},"id":"4220","type":"HoverTool"},{"attributes":{"axis_label":"Total number of draws","formatter":{"id":"4261"},"ticker":{"id":"4206"}},"id":"4205","type":"LinearAxis"},{"attributes":{},"id":"4206","type":"BasicTicker"},{"attributes":{"axis":{"id":"4205"},"ticker":null},"id":"4208","type":"Grid"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"4233","type":"Circle"},{"attributes":{},"id":"4269","type":"UnionRenderers"},{"attributes":{"toolbars":[{"id":"4223"}],"tools":[{"id":"4213"},{"id":"4214"},{"id":"4215"},{"id":"4216"},{"id":"4217"},{"id":"4218"},{"id":"4219"},{"id":"4220"}]},"id":"4275","type":"ProxyToolbar"},{"attributes":{},"id":"4270","type":"Selection"},{"attributes":{},"id":"4203","type":"LinearScale"},{"attributes":{"data_source":{"id":"4237"},"glyph":{"id":"4238"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4239"},"selection_glyph":null,"view":{"id":"4241"}},"id":"4240","type":"GlyphRenderer"},{"attributes":{"axis":{"id":"4209"},"dimension":1,"ticker":null},"id":"4212","type":"Grid"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"4222","type":"PolyAnnotation"},{"attributes":{},"id":"4210","type":"BasicTicker"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"4234","type":"Circle"},{"attributes":{"overlay":{"id":"4221"}},"id":"4215","type":"BoxZoomTool"},{"attributes":{},"id":"4214","type":"PanTool"},{"attributes":{"data":{"x":{"__ndarray__":"AAAAAAAAWUAAAAAAAABpQAAAAAAAwHJAAAAAAAAAeUAAAAAAAEB/QAAAAAAAwIJAAAAAAADghUAAAAAAAACJQAAAAAAAIIxAAAAAAABAj0AAAAAAADCRQAAAAAAAwJJAAAAAAABQlEAAAAAAAOCVQAAAAAAAcJdAAAAAAAAAmUAAAAAAAJCaQAAAAAAAIJxAAAAAAACwnUAAAAAAAECfQA==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"0glo/ZHQYUCWuV3JG5ppQMqT1P2iwXVAqY8DrPB3e0Bh+Z3qJ/WAQBkscwHQK4RAmBvOlEfWikBL3Hu21ASOQLfWD/YkmpBAzy5bwfaykkBL+YYwKf6TQNX3Ngx5b5VAciIDv7ZVlkBpY4usN1KYQKnVB/o5gplAPOgo6bOym0CHt7M3Mq+fQCfqDPZjtqBAJ784mgSeoUD/m8TvrdqhQA==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"4264"},"selection_policy":{"id":"4263"}},"id":"4232","type":"ColumnDataSource"},{"attributes":{},"id":"4213","type":"ResetTool"},{"attributes":{},"id":"4197","type":"DataRange1d"},{"attributes":{},"id":"4263","type":"UnionRenderers"},{"attributes":{},"id":"4219","type":"SaveTool"},{"attributes":{},"id":"4216","type":"WheelZoomTool"},{"attributes":{},"id":"4264","type":"Selection"},{"attributes":{"overlay":{"id":"4222"}},"id":"4217","type":"LassoSelectTool"},{"attributes":{"above":[{"id":"4253"}],"below":[{"id":"4205"}],"center":[{"id":"4208"},{"id":"4212"}],"left":[{"id":"4209"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"4235"},{"id":"4240"},{"id":"4245"},{"id":"4250"},{"id":"4252"}],"title":{"id":"4256"},"toolbar":{"id":"4223"},"toolbar_location":null,"x_range":{"id":"4197"},"x_scale":{"id":"4201"},"y_range":{"id":"4199"},"y_scale":{"id":"4203"}},"id":"4196","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"4218","type":"UndoTool"},{"attributes":{"data_source":{"id":"4242"},"glyph":{"id":"4243"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4244"},"selection_glyph":null,"view":{"id":"4246"}},"id":"4245","type":"GlyphRenderer"},{"attributes":{"line_color":"red","line_dash":[6],"line_width":3,"location":400},"id":"4252","type":"Span"},{"attributes":{},"id":"4259","type":"BasicTickFormatter"},{"attributes":{"data":{"x":{"__ndarray__":"AAAAAAAAWUAAAAAAAABpQAAAAAAAwHJAAAAAAAAAeUAAAAAAAEB/QAAAAAAAwIJAAAAAAADghUAAAAAAAACJQAAAAAAAIIxAAAAAAABAj0AAAAAAADCRQAAAAAAAwJJAAAAAAABQlEAAAAAAAOCVQAAAAAAAcJdAAAAAAAAAmUAAAAAAAJCaQAAAAAAAIJxAAAAAAACwnUAAAAAAAECfQA==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"kUgpITo5UkD/4xVCu9ZkQIwda8wmdG9AKVV7klfmc0B9iLN6f4t2QCN384Oug3lAXRIP57k5gEDe8oZZBOGEQCI3trf/S4dAdSETuLRMikCa5sM7ewSJQLpDDoqDVoxADogAvE7ZjUBXvDQTTveQQPztYWbX55BATSx1OrcnkUCJGFkHolCTQIz00+lP8ZNAUA/0blR0lEAOVbAHnSuWQA==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"4268"},"selection_policy":{"id":"4267"}},"id":"4242","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"orange"},"line_alpha":{"value":0.1},"line_color":{"value":"orange"},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"4249","type":"Circle"},{"attributes":{},"id":"4265","type":"UnionRenderers"},{"attributes":{"line_color":"orange","x":{"field":"x"},"y":{"field":"y"}},"id":"4243","type":"Line"},{"attributes":{"source":{"id":"4247"}},"id":"4251","type":"CDSView"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"4213"},{"id":"4214"},{"id":"4215"},{"id":"4216"},{"id":"4217"},{"id":"4218"},{"id":"4219"},{"id":"4220"}]},"id":"4223","type":"Toolbar"},{"attributes":{},"id":"4261","type":"BasicTickFormatter"},{"attributes":{"line_alpha":0.1,"line_color":"#1f77b4","x":{"field":"x"},"y":{"field":"y"}},"id":"4239","type":"Line"},{"attributes":{},"id":"4266","type":"Selection"},{"attributes":{"source":{"id":"4242"}},"id":"4246","type":"CDSView"},{"attributes":{"children":[[{"id":"4196"},0,0]]},"id":"4274","type":"GridBox"},{"attributes":{"source":{"id":"4232"}},"id":"4236","type":"CDSView"},{"attributes":{"source":{"id":"4237"}},"id":"4241","type":"CDSView"},{"attributes":{"line_color":"#1f77b4","x":{"field":"x"},"y":{"field":"y"}},"id":"4238","type":"Line"},{"attributes":{"line_alpha":0.1,"line_color":"orange","x":{"field":"x"},"y":{"field":"y"}},"id":"4244","type":"Line"},{"attributes":{"data":{"x":{"__ndarray__":"AAAAAAAAWUAAAAAAAABpQAAAAAAAwHJAAAAAAAAAeUAAAAAAAEB/QAAAAAAAwIJAAAAAAADghUAAAAAAAACJQAAAAAAAIIxAAAAAAABAj0AAAAAAADCRQAAAAAAAwJJAAAAAAABQlEAAAAAAAOCVQAAAAAAAcJdAAAAAAAAAmUAAAAAAAJCaQAAAAAAAIJxAAAAAAACwnUAAAAAAAECfQA==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"kUgpITo5UkD/4xVCu9ZkQIwda8wmdG9AKVV7klfmc0B9iLN6f4t2QCN384Oug3lAXRIP57k5gEDe8oZZBOGEQCI3trf/S4dAdSETuLRMikCa5sM7ewSJQLpDDoqDVoxADogAvE7ZjUBXvDQTTveQQPztYWbX55BATSx1OrcnkUCJGFkHolCTQIz00+lP8ZNAUA/0blR0lEAOVbAHnSuWQA==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"4270"},"selection_policy":{"id":"4269"}},"id":"4247","type":"ColumnDataSource"},{"attributes":{"data":{"x":{"__ndarray__":"AAAAAAAAWUAAAAAAAABpQAAAAAAAwHJAAAAAAAAAeUAAAAAAAEB/QAAAAAAAwIJAAAAAAADghUAAAAAAAACJQAAAAAAAIIxAAAAAAABAj0AAAAAAADCRQAAAAAAAwJJAAAAAAABQlEAAAAAAAOCVQAAAAAAAcJdAAAAAAAAAmUAAAAAAAJCaQAAAAAAAIJxAAAAAAACwnUAAAAAAAECfQA==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"0glo/ZHQYUCWuV3JG5ppQMqT1P2iwXVAqY8DrPB3e0Bh+Z3qJ/WAQBkscwHQK4RAmBvOlEfWikBL3Hu21ASOQLfWD/YkmpBAzy5bwfaykkBL+YYwKf6TQNX3Ngx5b5VAciIDv7ZVlkBpY4usN1KYQKnVB/o5gplAPOgo6bOym0CHt7M3Mq+fQCfqDPZjtqBAJ784mgSeoUD/m8TvrdqhQA==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"4266"},"selection_policy":{"id":"4265"}},"id":"4237","type":"ColumnDataSource"},{"attributes":{"fill_color":{"value":"orange"},"line_color":{"value":"orange"},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"4248","type":"Circle"},{"attributes":{"data_source":{"id":"4232"},"glyph":{"id":"4233"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4234"},"selection_glyph":null,"view":{"id":"4236"}},"id":"4235","type":"GlyphRenderer"},{"attributes":{},"id":"4199","type":"DataRange1d"},{"attributes":{},"id":"4267","type":"UnionRenderers"},{"attributes":{"label":{"value":"tail"},"renderers":[{"id":"4250"},{"id":"4245"}]},"id":"4255","type":"LegendItem"},{"attributes":{"label":{"value":"bulk"},"renderers":[{"id":"4235"},{"id":"4240"}]},"id":"4254","type":"LegendItem"},{"attributes":{},"id":"4268","type":"Selection"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"4221","type":"BoxAnnotation"},{"attributes":{"data_source":{"id":"4247"},"glyph":{"id":"4248"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4249"},"selection_glyph":null,"view":{"id":"4251"}},"id":"4250","type":"GlyphRenderer"},{"attributes":{"text":"b"},"id":"4256","type":"Title"},{"attributes":{},"id":"4201","type":"LinearScale"},{"attributes":{"click_policy":"hide","items":[{"id":"4254"},{"id":"4255"}],"location":"center_right","orientation":"horizontal"},"id":"4253","type":"Legend"}],"root_ids":["4277"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"f598b355-bd45-43d1-8c4b-292c852d4102","root_ids":["4277"],"roots":{"4277":"c1248422-ac16-4d25-90ac-4d7a79302184"}}];
                  root.Bokeh.embed.embed_items(docs_json, render_items);
                
                  }
                  if (root.Bokeh !== undefined) {
                    embed_document(root);
                  } else {
                    var attempts = 0;
                    var timer = setInterval(function(root) {
                      if (root.Bokeh !== undefined) {
                        clearInterval(timer);
                        embed_document(root);
                      } else {
                        attempts++;
                        if (attempts > 100) {
                          clearInterval(timer);
                          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                        }
                      }
                    }, 10, root)
                  }
                })(window);
              });
            };
            if (document.readyState != "loading") fn();
            else document.addEventListener("DOMContentLoaded", fn);
          })();
        },
        function(Bokeh) {
        
        
        }
      ];
    
      function run_inline_js() {
        
        for (var i = 0; i < inline_js.length; i++) {
          inline_js[i].call(root, root.Bokeh);
        }
        
      }
    
      if (root._bokeh_is_loading === 0) {
        console.debug("Bokeh: BokehJS loaded, going straight to plotting");
        run_inline_js();
      } else {
        load_libs(css_urls, js_urls, function() {
          console.debug("Bokeh: BokehJS plotting callback run at", now());
          run_inline_js();
        });
      }
    }(window));
  };
  if (document.readyState != "loading") fn();
  else document.addEventListener("DOMContentLoaded", fn);
})();