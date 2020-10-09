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
    
      
      
    
      var element = document.getElementById("d157f890-7413-43a5-bb54-21c8498cdcab");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid 'd157f890-7413-43a5-bb54-21c8498cdcab' but no matching script tag was found.")
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
                    
                  var docs_json = '{"e63fa66a-faff-4ee6-92c0-54512292543a":{"roots":{"references":[{"attributes":{"axis_label":"ELPD difference","formatter":{"id":"3999"},"ticker":{"id":"3970"}},"id":"3969","type":"LinearAxis"},{"attributes":{},"id":"3979","type":"SaveTool"},{"attributes":{"children":[[{"id":"3956"},0,0]]},"id":"4008","type":"GridBox"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"3973"},{"id":"3974"},{"id":"3975"},{"id":"3976"},{"id":"3977"},{"id":"3978"},{"id":"3979"},{"id":"3980"}]},"id":"3983","type":"Toolbar"},{"attributes":{"axis":{"id":"3969"},"dimension":1,"ticker":null},"id":"3972","type":"Grid"},{"attributes":{},"id":"4006","type":"UnionRenderers"},{"attributes":{},"id":"4005","type":"Selection"},{"attributes":{"toolbar":{"id":"4009"},"toolbar_location":"above"},"id":"4010","type":"ToolbarBox"},{"attributes":{},"id":"3957","type":"DataRange1d"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"3981","type":"BoxAnnotation"},{"attributes":{"line_color":{"value":"#1f77b4"},"size":{"field":"sizes","units":"screen"},"x":{"field":"xdata"},"y":{"field":"ydata"}},"id":"3992","type":"Cross"},{"attributes":{"toolbars":[{"id":"3983"}],"tools":[{"id":"3973"},{"id":"3974"},{"id":"3975"},{"id":"3976"},{"id":"3977"},{"id":"3978"},{"id":"3979"},{"id":"3980"}]},"id":"4009","type":"ProxyToolbar"},{"attributes":{"data":{"sizes":{"__ndarray__":"AAAAAAAAGEAAAAAAAAAYQAAAAAAAABhAAAAAAAAAGEAAAAAAAAAYQAAAAAAAABhAAAAAAAAAGEAAAAAAAAAYQA==","dtype":"float64","order":"little","shape":[8]},"xdata":[0,1,2,3,4,5,6,7],"ydata":{"__ndarray__":"gA887TQjqb8AatEMjdaMvwDAoFmzDUE/AIcKht05k7+A2X/x9IekPwDgrH2oaFM/ANQFB3wcsb8Am1vpFXuQvw==","dtype":"float64","order":"little","shape":[8]}},"selected":{"id":"4005"},"selection_policy":{"id":"4006"}},"id":"3993","type":"ColumnDataSource"},{"attributes":{"formatter":{"id":"4001"},"ticker":{"id":"3966"}},"id":"3965","type":"LinearAxis"},{"attributes":{"data_source":{"id":"3993"},"glyph":{"id":"3992"},"hover_glyph":null,"muted_glyph":null,"view":{"id":"3995"}},"id":"3994","type":"GlyphRenderer"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"3982","type":"PolyAnnotation"},{"attributes":{},"id":"3963","type":"LinearScale"},{"attributes":{"overlay":{"id":"3982"}},"id":"3977","type":"LassoSelectTool"},{"attributes":{"axis":{"id":"3965"},"ticker":null},"id":"3968","type":"Grid"},{"attributes":{},"id":"3976","type":"WheelZoomTool"},{"attributes":{"callback":null},"id":"3980","type":"HoverTool"},{"attributes":{},"id":"3973","type":"ResetTool"},{"attributes":{},"id":"3978","type":"UndoTool"},{"attributes":{"text":"Centered eight - Non centered eight"},"id":"3996","type":"Title"},{"attributes":{},"id":"3959","type":"DataRange1d"},{"attributes":{},"id":"3999","type":"BasicTickFormatter"},{"attributes":{},"id":"3970","type":"BasicTicker"},{"attributes":{"source":{"id":"3993"}},"id":"3995","type":"CDSView"},{"attributes":{},"id":"4001","type":"BasicTickFormatter"},{"attributes":{},"id":"3966","type":"BasicTicker"},{"attributes":{},"id":"3961","type":"LinearScale"},{"attributes":{},"id":"3974","type":"PanTool"},{"attributes":{"overlay":{"id":"3981"}},"id":"3975","type":"BoxZoomTool"},{"attributes":{"children":[{"id":"4010"},{"id":"4008"}]},"id":"4011","type":"Column"},{"attributes":{"below":[{"id":"3965"}],"center":[{"id":"3968"},{"id":"3972"}],"left":[{"id":"3969"}],"output_backend":"webgl","plot_height":288,"plot_width":384,"renderers":[{"id":"3994"}],"title":{"id":"3996"},"toolbar":{"id":"3983"},"toolbar_location":null,"x_range":{"id":"3957"},"x_scale":{"id":"3961"},"y_range":{"id":"3959"},"y_scale":{"id":"3963"}},"id":"3956","subtype":"Figure","type":"Plot"}],"root_ids":["4011"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"e63fa66a-faff-4ee6-92c0-54512292543a","root_ids":["4011"],"roots":{"4011":"d157f890-7413-43a5-bb54-21c8498cdcab"}}];
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