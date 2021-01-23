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
    
      
      
    
      var element = document.getElementById("49af5a80-cb2d-47fe-9e09-bb24700abbeb");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '49af5a80-cb2d-47fe-9e09-bb24700abbeb' but no matching script tag was found.")
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
                    
                  var docs_json = '{"f9d13de7-32d8-437b-aef3-3661ff2a6b82":{"roots":{"references":[{"attributes":{},"id":"17136","type":"UndoTool"},{"attributes":{"data_source":{"id":"17151"},"glyph":{"id":"17150"},"hover_glyph":null,"muted_glyph":null,"view":{"id":"17153"}},"id":"17152","type":"GlyphRenderer"},{"attributes":{},"id":"17128","type":"BasicTicker"},{"attributes":{"axis":{"id":"17127"},"dimension":1,"ticker":null},"id":"17130","type":"Grid"},{"attributes":{"overlay":{"id":"17139"}},"id":"17133","type":"BoxZoomTool"},{"attributes":{},"id":"17121","type":"LinearScale"},{"attributes":{},"id":"17134","type":"WheelZoomTool"},{"attributes":{"overlay":{"id":"17140"}},"id":"17135","type":"LassoSelectTool"},{"attributes":{},"id":"17137","type":"SaveTool"},{"attributes":{"callback":null},"id":"17138","type":"HoverTool"},{"attributes":{},"id":"17119","type":"LinearScale"},{"attributes":{"text":"Centered eight - Non centered eight"},"id":"17154","type":"Title"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"17139","type":"BoxAnnotation"},{"attributes":{},"id":"17115","type":"DataRange1d"},{"attributes":{},"id":"17132","type":"PanTool"},{"attributes":{},"id":"17162","type":"Selection"},{"attributes":{"children":[{"id":"17168"},{"id":"17166"}]},"id":"17169","type":"Column"},{"attributes":{},"id":"17117","type":"DataRange1d"},{"attributes":{"toolbars":[{"id":"17141"}],"tools":[{"id":"17131"},{"id":"17132"},{"id":"17133"},{"id":"17134"},{"id":"17135"},{"id":"17136"},{"id":"17137"},{"id":"17138"}]},"id":"17167","type":"ProxyToolbar"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"17140","type":"PolyAnnotation"},{"attributes":{},"id":"17124","type":"BasicTicker"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"17131"},{"id":"17132"},{"id":"17133"},{"id":"17134"},{"id":"17135"},{"id":"17136"},{"id":"17137"},{"id":"17138"}]},"id":"17141","type":"Toolbar"},{"attributes":{},"id":"17157","type":"BasicTickFormatter"},{"attributes":{},"id":"17161","type":"UnionRenderers"},{"attributes":{"data":{"sizes":{"__ndarray__":"AAAAAAAAGEAAAAAAAAAYQAAAAAAAABhAAAAAAAAAGEAAAAAAAAAYQAAAAAAAABhAAAAAAAAAGEAAAAAAAAAYQA==","dtype":"float64","order":"little","shape":[8]},"xdata":[0,1,2,3,4,5,6,7],"ydata":{"__ndarray__":"gA887TQjqb8AatEMjdaMvwDAoFmzDUE/AIcKht05k7+A2X/x9IekPwDgrH2oaFM/ANQFB3wcsb8Am1vpFXuQvw==","dtype":"float64","order":"little","shape":[8]}},"selected":{"id":"17162"},"selection_policy":{"id":"17161"}},"id":"17151","type":"ColumnDataSource"},{"attributes":{},"id":"17159","type":"BasicTickFormatter"},{"attributes":{"axis":{"id":"17123"},"ticker":null},"id":"17126","type":"Grid"},{"attributes":{"axis_label":"ELPD difference","formatter":{"id":"17159"},"ticker":{"id":"17128"}},"id":"17127","type":"LinearAxis"},{"attributes":{"line_color":{"value":"#2a2eec"},"size":{"field":"sizes","units":"screen"},"x":{"field":"xdata"},"y":{"field":"ydata"}},"id":"17150","type":"Cross"},{"attributes":{"toolbar":{"id":"17167"},"toolbar_location":"above"},"id":"17168","type":"ToolbarBox"},{"attributes":{"source":{"id":"17151"}},"id":"17153","type":"CDSView"},{"attributes":{"below":[{"id":"17123"}],"center":[{"id":"17126"},{"id":"17130"}],"left":[{"id":"17127"}],"output_backend":"webgl","plot_height":288,"plot_width":432,"renderers":[{"id":"17152"}],"title":{"id":"17154"},"toolbar":{"id":"17141"},"toolbar_location":null,"x_range":{"id":"17115"},"x_scale":{"id":"17119"},"y_range":{"id":"17117"},"y_scale":{"id":"17121"}},"id":"17114","subtype":"Figure","type":"Plot"},{"attributes":{"children":[[{"id":"17114"},0,0]]},"id":"17166","type":"GridBox"},{"attributes":{"formatter":{"id":"17157"},"ticker":{"id":"17124"}},"id":"17123","type":"LinearAxis"},{"attributes":{},"id":"17131","type":"ResetTool"}],"root_ids":["17169"]},"title":"Bokeh Application","version":"2.2.3"}}';
                  var render_items = [{"docid":"f9d13de7-32d8-437b-aef3-3661ff2a6b82","root_ids":["17169"],"roots":{"17169":"49af5a80-cb2d-47fe-9e09-bb24700abbeb"}}];
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