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
    
      
      
    
      var element = document.getElementById("6c9cf563-5904-491d-ae63-16151e39fc1f");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '6c9cf563-5904-491d-ae63-16151e39fc1f' but no matching script tag was found.")
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
    
        const hashes = {"https://cdn.bokeh.org/bokeh/release/bokeh-2.2.3.min.js": "T2yuo9Oe71Cz/I4X9Ac5+gpEa5a8PpJCDlqKYO0CfAuEszu1JrXLl8YugMqYe3sM", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.3.min.js": "98GDGJ0kOMCUMUePhksaQ/GYgB3+NH9h996V88sh3aOiUNX3N+fLXAtry6xctSZ6", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.3.min.js": "89bArO+nlbP3sgakeHjCo1JYxYR5wufVgA3IbUvDY+K7w4zyxJqssu7wVnfeKCq8"};
    
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
    
      
      var js_urls = ["https://cdn.bokeh.org/bokeh/release/bokeh-2.2.3.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.3.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.3.min.js"];
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
                    
                  var docs_json = '{"901a916e-9208-423d-bc0c-d05858df3457":{"roots":{"references":[{"attributes":{"axis":{"id":"17363"},"ticker":null},"id":"17366","type":"Grid"},{"attributes":{},"id":"17417","type":"BasicTickFormatter"},{"attributes":{"overlay":{"id":"17379"}},"id":"17373","type":"BoxZoomTool"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"orange"},"line_alpha":{"value":0.1},"line_color":{"value":"orange"},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"17407","type":"Circle"},{"attributes":{"data":{"x":{"__ndarray__":"AAAAAAAAWUAAAAAAAABpQAAAAAAAwHJAAAAAAAAAeUAAAAAAAEB/QAAAAAAAwIJAAAAAAADghUAAAAAAAACJQAAAAAAAIIxAAAAAAABAj0AAAAAAADCRQAAAAAAAwJJAAAAAAABQlEAAAAAAAOCVQAAAAAAAcJdAAAAAAAAAmUAAAAAAAJCaQAAAAAAAIJxAAAAAAACwnUAAAAAAAECfQA==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"kUgpITo5UkD/4xVCu9ZkQIwda8wmdG9AKVV7klfmc0B9iLN6f4t2QCN384Oug3lAXRIP57k5gEDe8oZZBOGEQCI3trf/S4dAdSETuLRMikCa5sM7ewSJQLpDDoqDVoxADogAvE7ZjUBXvDQTTveQQPztYWbX55BATSx1OrcnkUCJGFkHolCTQIz00+lP8ZNAUA/0blR0lEAOVbAHnSuWQA==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"17426"},"selection_policy":{"id":"17425"}},"id":"17400","type":"ColumnDataSource"},{"attributes":{"text":"b"},"id":"17414","type":"Title"},{"attributes":{"data":{"x":{"__ndarray__":"AAAAAAAAWUAAAAAAAABpQAAAAAAAwHJAAAAAAAAAeUAAAAAAAEB/QAAAAAAAwIJAAAAAAADghUAAAAAAAACJQAAAAAAAIIxAAAAAAABAj0AAAAAAADCRQAAAAAAAwJJAAAAAAABQlEAAAAAAAOCVQAAAAAAAcJdAAAAAAAAAmUAAAAAAAJCaQAAAAAAAIJxAAAAAAACwnUAAAAAAAECfQA==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"0glo/ZHQYUCWuV3JG5ppQMqT1P2iwXVAqY8DrPB3e0Bh+Z3qJ/WAQBkscwHQK4RAmBvOlEfWikBL3Hu21ASOQLfWD/YkmpBAzy5bwfaykkBL+YYwKf6TQNX3Ngx5b5VAciIDv7ZVlkBpY4usN1KYQKnVB/o5gplAPOgo6bOym0CHt7M3Mq+fQCfqDPZjtqBAJ784mgSeoUD/m8TvrdqhQA==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"17422"},"selection_policy":{"id":"17421"}},"id":"17390","type":"ColumnDataSource"},{"attributes":{"axis":{"id":"17367"},"dimension":1,"ticker":null},"id":"17370","type":"Grid"},{"attributes":{"axis_label":"Total number of draws","formatter":{"id":"17417"},"ticker":{"id":"17364"}},"id":"17363","type":"LinearAxis"},{"attributes":{},"id":"17419","type":"BasicTickFormatter"},{"attributes":{"children":[{"id":"17434"},{"id":"17432"}]},"id":"17435","type":"Column"},{"attributes":{},"id":"17371","type":"ResetTool"},{"attributes":{},"id":"17422","type":"Selection"},{"attributes":{},"id":"17425","type":"UnionRenderers"},{"attributes":{"line_alpha":0.1,"line_color":"orange","x":{"field":"x"},"y":{"field":"y"}},"id":"17402","type":"Line"},{"attributes":{},"id":"17423","type":"UnionRenderers"},{"attributes":{"above":[{"id":"17411"}],"below":[{"id":"17363"}],"center":[{"id":"17366"},{"id":"17370"}],"left":[{"id":"17367"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"17393"},{"id":"17398"},{"id":"17403"},{"id":"17408"},{"id":"17410"}],"title":{"id":"17414"},"toolbar":{"id":"17381"},"toolbar_location":null,"x_range":{"id":"17355"},"x_scale":{"id":"17359"},"y_range":{"id":"17357"},"y_scale":{"id":"17361"}},"id":"17354","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"17427","type":"UnionRenderers"},{"attributes":{"label":{"value":"tail"},"renderers":[{"id":"17408"},{"id":"17403"}]},"id":"17413","type":"LegendItem"},{"attributes":{"fill_color":{"value":"orange"},"line_color":{"value":"orange"},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"17406","type":"Circle"},{"attributes":{"data_source":{"id":"17400"},"glyph":{"id":"17401"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17402"},"selection_glyph":null,"view":{"id":"17404"}},"id":"17403","type":"GlyphRenderer"},{"attributes":{"toolbars":[{"id":"17381"}],"tools":[{"id":"17371"},{"id":"17372"},{"id":"17373"},{"id":"17374"},{"id":"17375"},{"id":"17376"},{"id":"17377"},{"id":"17378"}]},"id":"17433","type":"ProxyToolbar"},{"attributes":{"axis_label":"ESS","formatter":{"id":"17419"},"ticker":{"id":"17368"}},"id":"17367","type":"LinearAxis"},{"attributes":{},"id":"17364","type":"BasicTicker"},{"attributes":{},"id":"17424","type":"Selection"},{"attributes":{"click_policy":"hide","items":[{"id":"17412"},{"id":"17413"}],"location":"center_right","orientation":"horizontal"},"id":"17411","type":"Legend"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"17371"},{"id":"17372"},{"id":"17373"},{"id":"17374"},{"id":"17375"},{"id":"17376"},{"id":"17377"},{"id":"17378"}]},"id":"17381","type":"Toolbar"},{"attributes":{"line_alpha":0.1,"line_color":"#1f77b4","x":{"field":"x"},"y":{"field":"y"}},"id":"17397","type":"Line"},{"attributes":{"data_source":{"id":"17390"},"glyph":{"id":"17391"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17392"},"selection_glyph":null,"view":{"id":"17394"}},"id":"17393","type":"GlyphRenderer"},{"attributes":{},"id":"17357","type":"DataRange1d"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"17379","type":"BoxAnnotation"},{"attributes":{"line_color":"red","line_dash":[6],"line_width":3,"location":400},"id":"17410","type":"Span"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"17380","type":"PolyAnnotation"},{"attributes":{"label":{"value":"bulk"},"renderers":[{"id":"17393"},{"id":"17398"}]},"id":"17412","type":"LegendItem"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"17391","type":"Circle"},{"attributes":{"line_color":"#1f77b4","x":{"field":"x"},"y":{"field":"y"}},"id":"17396","type":"Line"},{"attributes":{"children":[[{"id":"17354"},0,0]]},"id":"17432","type":"GridBox"},{"attributes":{"source":{"id":"17395"}},"id":"17399","type":"CDSView"},{"attributes":{"overlay":{"id":"17380"}},"id":"17375","type":"LassoSelectTool"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"17392","type":"Circle"},{"attributes":{},"id":"17374","type":"WheelZoomTool"},{"attributes":{"data_source":{"id":"17405"},"glyph":{"id":"17406"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17407"},"selection_glyph":null,"view":{"id":"17409"}},"id":"17408","type":"GlyphRenderer"},{"attributes":{"source":{"id":"17390"}},"id":"17394","type":"CDSView"},{"attributes":{},"id":"17372","type":"PanTool"},{"attributes":{"source":{"id":"17405"}},"id":"17409","type":"CDSView"},{"attributes":{"callback":null},"id":"17378","type":"HoverTool"},{"attributes":{"data":{"x":{"__ndarray__":"AAAAAAAAWUAAAAAAAABpQAAAAAAAwHJAAAAAAAAAeUAAAAAAAEB/QAAAAAAAwIJAAAAAAADghUAAAAAAAACJQAAAAAAAIIxAAAAAAABAj0AAAAAAADCRQAAAAAAAwJJAAAAAAABQlEAAAAAAAOCVQAAAAAAAcJdAAAAAAAAAmUAAAAAAAJCaQAAAAAAAIJxAAAAAAACwnUAAAAAAAECfQA==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"0glo/ZHQYUCWuV3JG5ppQMqT1P2iwXVAqY8DrPB3e0Bh+Z3qJ/WAQBkscwHQK4RAmBvOlEfWikBL3Hu21ASOQLfWD/YkmpBAzy5bwfaykkBL+YYwKf6TQNX3Ngx5b5VAciIDv7ZVlkBpY4usN1KYQKnVB/o5gplAPOgo6bOym0CHt7M3Mq+fQCfqDPZjtqBAJ784mgSeoUD/m8TvrdqhQA==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"17424"},"selection_policy":{"id":"17423"}},"id":"17395","type":"ColumnDataSource"},{"attributes":{},"id":"17376","type":"UndoTool"},{"attributes":{"line_color":"orange","x":{"field":"x"},"y":{"field":"y"}},"id":"17401","type":"Line"},{"attributes":{"data":{"x":{"__ndarray__":"AAAAAAAAWUAAAAAAAABpQAAAAAAAwHJAAAAAAAAAeUAAAAAAAEB/QAAAAAAAwIJAAAAAAADghUAAAAAAAACJQAAAAAAAIIxAAAAAAABAj0AAAAAAADCRQAAAAAAAwJJAAAAAAABQlEAAAAAAAOCVQAAAAAAAcJdAAAAAAAAAmUAAAAAAAJCaQAAAAAAAIJxAAAAAAACwnUAAAAAAAECfQA==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"kUgpITo5UkD/4xVCu9ZkQIwda8wmdG9AKVV7klfmc0B9iLN6f4t2QCN384Oug3lAXRIP57k5gEDe8oZZBOGEQCI3trf/S4dAdSETuLRMikCa5sM7ewSJQLpDDoqDVoxADogAvE7ZjUBXvDQTTveQQPztYWbX55BATSx1OrcnkUCJGFkHolCTQIz00+lP8ZNAUA/0blR0lEAOVbAHnSuWQA==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"17428"},"selection_policy":{"id":"17427"}},"id":"17405","type":"ColumnDataSource"},{"attributes":{},"id":"17368","type":"BasicTicker"},{"attributes":{},"id":"17355","type":"DataRange1d"},{"attributes":{},"id":"17428","type":"Selection"},{"attributes":{},"id":"17421","type":"UnionRenderers"},{"attributes":{},"id":"17359","type":"LinearScale"},{"attributes":{},"id":"17377","type":"SaveTool"},{"attributes":{},"id":"17426","type":"Selection"},{"attributes":{},"id":"17361","type":"LinearScale"},{"attributes":{"toolbar":{"id":"17433"},"toolbar_location":"above"},"id":"17434","type":"ToolbarBox"},{"attributes":{"data_source":{"id":"17395"},"glyph":{"id":"17396"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17397"},"selection_glyph":null,"view":{"id":"17399"}},"id":"17398","type":"GlyphRenderer"},{"attributes":{"source":{"id":"17400"}},"id":"17404","type":"CDSView"}],"root_ids":["17435"]},"title":"Bokeh Application","version":"2.2.3"}}';
                  var render_items = [{"docid":"901a916e-9208-423d-bc0c-d05858df3457","root_ids":["17435"],"roots":{"17435":"6c9cf563-5904-491d-ae63-16151e39fc1f"}}];
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