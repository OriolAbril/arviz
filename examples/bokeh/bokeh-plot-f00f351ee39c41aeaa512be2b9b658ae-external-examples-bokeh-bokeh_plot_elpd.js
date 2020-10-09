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
    
      
      
    
      var element = document.getElementById("d1393cbd-f7d6-447c-8c87-438cf1059377");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid 'd1393cbd-f7d6-447c-8c87-438cf1059377' but no matching script tag was found.")
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
                    
                  var docs_json = '{"19401736-7d14-4de3-b843-6595f52064ae":{"roots":{"references":[{"attributes":{},"id":"3865","type":"DataRange1d"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"3890","type":"PolyAnnotation"},{"attributes":{"children":[{"id":"3918"},{"id":"3916"}]},"id":"3919","type":"Column"},{"attributes":{"data":{"sizes":{"__ndarray__":"AAAAAAAAGEAAAAAAAAAYQAAAAAAAABhAAAAAAAAAGEAAAAAAAAAYQAAAAAAAABhAAAAAAAAAGEAAAAAAAAAYQA==","dtype":"float64","order":"little","shape":[8]},"xdata":[0,1,2,3,4,5,6,7],"ydata":{"__ndarray__":"gA887TQjqb8AatEMjdaMvwDAoFmzDUE/AIcKht05k7+A2X/x9IekPwDgrH2oaFM/ANQFB3wcsb8Am1vpFXuQvw==","dtype":"float64","order":"little","shape":[8]}},"selected":{"id":"3911"},"selection_policy":{"id":"3912"}},"id":"3901","type":"ColumnDataSource"},{"attributes":{"axis":{"id":"3877"},"dimension":1,"ticker":null},"id":"3880","type":"Grid"},{"attributes":{},"id":"3909","type":"BasicTickFormatter"},{"attributes":{},"id":"3869","type":"LinearScale"},{"attributes":{},"id":"3878","type":"BasicTicker"},{"attributes":{},"id":"3907","type":"BasicTickFormatter"},{"attributes":{"callback":null},"id":"3888","type":"HoverTool"},{"attributes":{},"id":"3871","type":"LinearScale"},{"attributes":{"data_source":{"id":"3901"},"glyph":{"id":"3900"},"hover_glyph":null,"muted_glyph":null,"view":{"id":"3903"}},"id":"3902","type":"GlyphRenderer"},{"attributes":{"formatter":{"id":"3909"},"ticker":{"id":"3874"}},"id":"3873","type":"LinearAxis"},{"attributes":{},"id":"3887","type":"SaveTool"},{"attributes":{},"id":"3874","type":"BasicTicker"},{"attributes":{"toolbars":[{"id":"3891"}],"tools":[{"id":"3881"},{"id":"3882"},{"id":"3883"},{"id":"3884"},{"id":"3885"},{"id":"3886"},{"id":"3887"},{"id":"3888"}]},"id":"3917","type":"ProxyToolbar"},{"attributes":{},"id":"3867","type":"DataRange1d"},{"attributes":{},"id":"3911","type":"Selection"},{"attributes":{},"id":"3912","type":"UnionRenderers"},{"attributes":{"source":{"id":"3901"}},"id":"3903","type":"CDSView"},{"attributes":{"axis_label":"ELPD difference","formatter":{"id":"3907"},"ticker":{"id":"3878"}},"id":"3877","type":"LinearAxis"},{"attributes":{},"id":"3886","type":"UndoTool"},{"attributes":{"overlay":{"id":"3890"}},"id":"3885","type":"LassoSelectTool"},{"attributes":{"text":"Centered eight - Non centered eight"},"id":"3904","type":"Title"},{"attributes":{},"id":"3884","type":"WheelZoomTool"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"3889","type":"BoxAnnotation"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"3881"},{"id":"3882"},{"id":"3883"},{"id":"3884"},{"id":"3885"},{"id":"3886"},{"id":"3887"},{"id":"3888"}]},"id":"3891","type":"Toolbar"},{"attributes":{"axis":{"id":"3873"},"ticker":null},"id":"3876","type":"Grid"},{"attributes":{"overlay":{"id":"3889"}},"id":"3883","type":"BoxZoomTool"},{"attributes":{"line_color":{"value":"#1f77b4"},"size":{"field":"sizes","units":"screen"},"x":{"field":"xdata"},"y":{"field":"ydata"}},"id":"3900","type":"Cross"},{"attributes":{"toolbar":{"id":"3917"},"toolbar_location":"above"},"id":"3918","type":"ToolbarBox"},{"attributes":{"children":[[{"id":"3864"},0,0]]},"id":"3916","type":"GridBox"},{"attributes":{},"id":"3882","type":"PanTool"},{"attributes":{},"id":"3881","type":"ResetTool"},{"attributes":{"below":[{"id":"3873"}],"center":[{"id":"3876"},{"id":"3880"}],"left":[{"id":"3877"}],"output_backend":"webgl","plot_height":288,"plot_width":384,"renderers":[{"id":"3902"}],"title":{"id":"3904"},"toolbar":{"id":"3891"},"toolbar_location":null,"x_range":{"id":"3865"},"x_scale":{"id":"3869"},"y_range":{"id":"3867"},"y_scale":{"id":"3871"}},"id":"3864","subtype":"Figure","type":"Plot"}],"root_ids":["3919"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"19401736-7d14-4de3-b843-6595f52064ae","root_ids":["3919"],"roots":{"3919":"d1393cbd-f7d6-447c-8c87-438cf1059377"}}];
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