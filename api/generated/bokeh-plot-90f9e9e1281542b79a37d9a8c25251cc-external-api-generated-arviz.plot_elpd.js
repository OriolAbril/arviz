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
    
      
      
    
      var element = document.getElementById("3c686131-3f1d-4881-9291-dcfbb7543586");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '3c686131-3f1d-4881-9291-dcfbb7543586' but no matching script tag was found.")
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
                    
                  var docs_json = '{"721a63e1-19e8-4a40-ad88-5192b139ed00":{"roots":{"references":[{"attributes":{"overlay":{"id":"1027"}},"id":"1022","type":"LassoSelectTool"},{"attributes":{},"id":"1006","type":"LinearScale"},{"attributes":{"children":[[{"id":"1001"},0,0]]},"id":"1053","type":"GridBox"},{"attributes":{},"id":"1050","type":"UnionRenderers"},{"attributes":{"toolbars":[{"id":"1028"}],"tools":[{"id":"1018"},{"id":"1019"},{"id":"1020"},{"id":"1021"},{"id":"1022"},{"id":"1023"},{"id":"1024"},{"id":"1025"}]},"id":"1054","type":"ProxyToolbar"},{"attributes":{},"id":"1002","type":"DataRange1d"},{"attributes":{},"id":"1024","type":"SaveTool"},{"attributes":{"text":"centered model - non centered model"},"id":"1041","type":"Title"},{"attributes":{"callback":null},"id":"1025","type":"HoverTool"},{"attributes":{"source":{"id":"1038"}},"id":"1040","type":"CDSView"},{"attributes":{},"id":"1019","type":"PanTool"},{"attributes":{},"id":"1011","type":"BasicTicker"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"1027","type":"PolyAnnotation"},{"attributes":{"axis":{"id":"1010"},"ticker":null},"id":"1013","type":"Grid"},{"attributes":{"children":[{"id":"1055"},{"id":"1053"}]},"id":"1056","type":"Column"},{"attributes":{"data_source":{"id":"1038"},"glyph":{"id":"1037"},"hover_glyph":null,"muted_glyph":null,"view":{"id":"1040"}},"id":"1039","type":"GlyphRenderer"},{"attributes":{},"id":"1023","type":"UndoTool"},{"attributes":{"overlay":{"id":"1026"}},"id":"1020","type":"BoxZoomTool"},{"attributes":{},"id":"1021","type":"WheelZoomTool"},{"attributes":{},"id":"1004","type":"DataRange1d"},{"attributes":{"below":[{"id":"1010"}],"center":[{"id":"1013"},{"id":"1017"}],"left":[{"id":"1014"}],"output_backend":"webgl","plot_height":288,"plot_width":384,"renderers":[{"id":"1039"}],"title":{"id":"1041"},"toolbar":{"id":"1028"},"toolbar_location":null,"x_range":{"id":"1002"},"x_scale":{"id":"1006"},"y_range":{"id":"1004"},"y_scale":{"id":"1008"}},"id":"1001","subtype":"Figure","type":"Plot"},{"attributes":{"toolbar":{"id":"1054"},"toolbar_location":"above"},"id":"1055","type":"ToolbarBox"},{"attributes":{"line_color":{"value":"#1f77b4"},"size":{"field":"sizes","units":"screen"},"x":{"field":"xdata"},"y":{"field":"ydata"}},"id":"1037","type":"Cross"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"1018"},{"id":"1019"},{"id":"1020"},{"id":"1021"},{"id":"1022"},{"id":"1023"},{"id":"1024"},{"id":"1025"}]},"id":"1028","type":"Toolbar"},{"attributes":{},"id":"1046","type":"BasicTickFormatter"},{"attributes":{},"id":"1018","type":"ResetTool"},{"attributes":{},"id":"1015","type":"BasicTicker"},{"attributes":{"axis":{"id":"1014"},"dimension":1,"ticker":null},"id":"1017","type":"Grid"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"1026","type":"BoxAnnotation"},{"attributes":{},"id":"1051","type":"Selection"},{"attributes":{"formatter":{"id":"1046"},"ticker":{"id":"1011"}},"id":"1010","type":"LinearAxis"},{"attributes":{"axis_label":"ELPD difference","formatter":{"id":"1044"},"ticker":{"id":"1015"}},"id":"1014","type":"LinearAxis"},{"attributes":{"data":{"sizes":{"__ndarray__":"AAAAAAAAGEAAAAAAAAAYQAAAAAAAABhAAAAAAAAAGEAAAAAAAAAYQAAAAAAAABhAAAAAAAAAGEAAAAAAAAAYQA==","dtype":"float64","order":"little","shape":[8]},"xdata":[0,1,2,3,4,5,6,7],"ydata":{"__ndarray__":"gA887TQjqb8AatEMjdaMvwDAoFmzDUE/AIcKht05k7+A2X/x9IekPwDgrH2oaFM/ANQFB3wcsb8Am1vpFXuQvw==","dtype":"float64","order":"little","shape":[8]}},"selected":{"id":"1051"},"selection_policy":{"id":"1050"}},"id":"1038","type":"ColumnDataSource"},{"attributes":{},"id":"1008","type":"LinearScale"},{"attributes":{},"id":"1044","type":"BasicTickFormatter"}],"root_ids":["1056"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"721a63e1-19e8-4a40-ad88-5192b139ed00","root_ids":["1056"],"roots":{"1056":"3c686131-3f1d-4881-9291-dcfbb7543586"}}];
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