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
    
      
      
    
      var element = document.getElementById("d2be7380-dea5-4d37-b54f-9800e16fa8e5");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid 'd2be7380-dea5-4d37-b54f-9800e16fa8e5' but no matching script tag was found.")
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
                    
                  var docs_json = '{"ca623b54-db3e-4e32-98eb-6f39875c1a36":{"roots":{"references":[{"attributes":{"formatter":{"id":"17056"},"ticker":{"id":"17022"}},"id":"17021","type":"LinearAxis"},{"attributes":{},"id":"17015","type":"DataRange1d"},{"attributes":{},"id":"17034","type":"UndoTool"},{"attributes":{},"id":"17056","type":"BasicTickFormatter"},{"attributes":{},"id":"17022","type":"BasicTicker"},{"attributes":{"below":[{"id":"17021"}],"center":[{"id":"17024"},{"id":"17028"}],"left":[{"id":"17025"}],"output_backend":"webgl","plot_height":288,"plot_width":432,"renderers":[{"id":"17050"}],"title":{"id":"17052"},"toolbar":{"id":"17039"},"toolbar_location":null,"x_range":{"id":"17013"},"x_scale":{"id":"17017"},"y_range":{"id":"17015"},"y_scale":{"id":"17019"}},"id":"17012","subtype":"Figure","type":"Plot"},{"attributes":{"data":{"sizes":{"__ndarray__":"AAAAAAAAGEAAAAAAAAAYQAAAAAAAABhAAAAAAAAAGEAAAAAAAAAYQAAAAAAAABhAAAAAAAAAGEAAAAAAAAAYQA==","dtype":"float64","order":"little","shape":[8]},"xdata":[0,1,2,3,4,5,6,7],"ydata":{"__ndarray__":"gA887TQjqb8AatEMjdaMvwDAoFmzDUE/AIcKht05k7+A2X/x9IekPwDgrH2oaFM/ANQFB3wcsb8Am1vpFXuQvw==","dtype":"float64","order":"little","shape":[8]}},"selected":{"id":"17062"},"selection_policy":{"id":"17061"}},"id":"17049","type":"ColumnDataSource"},{"attributes":{"axis":{"id":"17021"},"ticker":null},"id":"17024","type":"Grid"},{"attributes":{},"id":"17061","type":"UnionRenderers"},{"attributes":{"line_color":{"value":"#2a2eec"},"size":{"field":"sizes","units":"screen"},"x":{"field":"xdata"},"y":{"field":"ydata"}},"id":"17048","type":"Cross"},{"attributes":{"overlay":{"id":"17037"}},"id":"17031","type":"BoxZoomTool"},{"attributes":{},"id":"17062","type":"Selection"},{"attributes":{},"id":"17032","type":"WheelZoomTool"},{"attributes":{"children":[[{"id":"17012"},0,0]]},"id":"17064","type":"GridBox"},{"attributes":{},"id":"17026","type":"BasicTicker"},{"attributes":{},"id":"17029","type":"ResetTool"},{"attributes":{"data_source":{"id":"17049"},"glyph":{"id":"17048"},"hover_glyph":null,"muted_glyph":null,"view":{"id":"17051"}},"id":"17050","type":"GlyphRenderer"},{"attributes":{"overlay":{"id":"17038"}},"id":"17033","type":"LassoSelectTool"},{"attributes":{"text":"Centered eight - Non centered eight"},"id":"17052","type":"Title"},{"attributes":{},"id":"17019","type":"LinearScale"},{"attributes":{},"id":"17030","type":"PanTool"},{"attributes":{"callback":null},"id":"17036","type":"HoverTool"},{"attributes":{"children":[{"id":"17066"},{"id":"17064"}]},"id":"17067","type":"Column"},{"attributes":{},"id":"17017","type":"LinearScale"},{"attributes":{},"id":"17013","type":"DataRange1d"},{"attributes":{"axis":{"id":"17025"},"dimension":1,"ticker":null},"id":"17028","type":"Grid"},{"attributes":{},"id":"17035","type":"SaveTool"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"17029"},{"id":"17030"},{"id":"17031"},{"id":"17032"},{"id":"17033"},{"id":"17034"},{"id":"17035"},{"id":"17036"}]},"id":"17039","type":"Toolbar"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"17038","type":"PolyAnnotation"},{"attributes":{},"id":"17058","type":"BasicTickFormatter"},{"attributes":{"axis_label":"ELPD difference","formatter":{"id":"17058"},"ticker":{"id":"17026"}},"id":"17025","type":"LinearAxis"},{"attributes":{"toolbar":{"id":"17065"},"toolbar_location":"above"},"id":"17066","type":"ToolbarBox"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"17037","type":"BoxAnnotation"},{"attributes":{"source":{"id":"17049"}},"id":"17051","type":"CDSView"},{"attributes":{"toolbars":[{"id":"17039"}],"tools":[{"id":"17029"},{"id":"17030"},{"id":"17031"},{"id":"17032"},{"id":"17033"},{"id":"17034"},{"id":"17035"},{"id":"17036"}]},"id":"17065","type":"ProxyToolbar"}],"root_ids":["17067"]},"title":"Bokeh Application","version":"2.2.3"}}';
                  var render_items = [{"docid":"ca623b54-db3e-4e32-98eb-6f39875c1a36","root_ids":["17067"],"roots":{"17067":"d2be7380-dea5-4d37-b54f-9800e16fa8e5"}}];
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