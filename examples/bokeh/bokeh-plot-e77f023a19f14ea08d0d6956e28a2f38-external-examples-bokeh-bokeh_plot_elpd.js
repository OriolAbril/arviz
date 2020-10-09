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
    
      
      
    
      var element = document.getElementById("e1f639f2-dc86-4993-b4f1-3f2c0dddd30c");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid 'e1f639f2-dc86-4993-b4f1-3f2c0dddd30c' but no matching script tag was found.")
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
                    
                  var docs_json = '{"6bab5c6d-77b0-4434-95f4-f4e714e757c6":{"roots":{"references":[{"attributes":{},"id":"67749","type":"SaveTool"},{"attributes":{},"id":"67774","type":"UnionRenderers"},{"attributes":{},"id":"67773","type":"Selection"},{"attributes":{"axis":{"id":"67739"},"dimension":1,"ticker":null},"id":"67742","type":"Grid"},{"attributes":{"toolbars":[{"id":"67753"}],"tools":[{"id":"67743"},{"id":"67744"},{"id":"67745"},{"id":"67746"},{"id":"67747"},{"id":"67748"},{"id":"67749"},{"id":"67750"}]},"id":"67779","type":"ProxyToolbar"},{"attributes":{"overlay":{"id":"67752"}},"id":"67747","type":"LassoSelectTool"},{"attributes":{"overlay":{"id":"67751"}},"id":"67745","type":"BoxZoomTool"},{"attributes":{"toolbar":{"id":"67779"},"toolbar_location":"above"},"id":"67780","type":"ToolbarBox"},{"attributes":{"data_source":{"id":"67763"},"glyph":{"id":"67762"},"hover_glyph":null,"muted_glyph":null,"view":{"id":"67765"}},"id":"67764","type":"GlyphRenderer"},{"attributes":{"callback":null},"id":"67750","type":"HoverTool"},{"attributes":{"data":{"sizes":{"__ndarray__":"AAAAAAAAGEAAAAAAAAAYQAAAAAAAABhAAAAAAAAAGEAAAAAAAAAYQAAAAAAAABhAAAAAAAAAGEAAAAAAAAAYQA==","dtype":"float64","order":"little","shape":[8]},"xdata":[0,1,2,3,4,5,6,7],"ydata":{"__ndarray__":"gA887TQjqb8AatEMjdaMvwDAoFmzDUE/AIcKht05k7+A2X/x9IekPwDgrH2oaFM/ANQFB3wcsb8Am1vpFXuQvw==","dtype":"float64","order":"little","shape":[8]}},"selected":{"id":"67773"},"selection_policy":{"id":"67774"}},"id":"67763","type":"ColumnDataSource"},{"attributes":{},"id":"67746","type":"WheelZoomTool"},{"attributes":{"line_color":{"value":"#2a2eec"},"size":{"field":"sizes","units":"screen"},"x":{"field":"xdata"},"y":{"field":"ydata"}},"id":"67762","type":"Cross"},{"attributes":{},"id":"67731","type":"LinearScale"},{"attributes":{},"id":"67769","type":"BasicTickFormatter"},{"attributes":{"axis":{"id":"67735"},"ticker":null},"id":"67738","type":"Grid"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"67751","type":"BoxAnnotation"},{"attributes":{"below":[{"id":"67735"}],"center":[{"id":"67738"},{"id":"67742"}],"left":[{"id":"67739"}],"output_backend":"webgl","plot_height":288,"plot_width":432,"renderers":[{"id":"67764"}],"title":{"id":"67766"},"toolbar":{"id":"67753"},"toolbar_location":null,"x_range":{"id":"67727"},"x_scale":{"id":"67731"},"y_range":{"id":"67729"},"y_scale":{"id":"67733"}},"id":"67726","subtype":"Figure","type":"Plot"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"67743"},{"id":"67744"},{"id":"67745"},{"id":"67746"},{"id":"67747"},{"id":"67748"},{"id":"67749"},{"id":"67750"}]},"id":"67753","type":"Toolbar"},{"attributes":{},"id":"67743","type":"ResetTool"},{"attributes":{"children":[{"id":"67780"},{"id":"67778"}]},"id":"67781","type":"Column"},{"attributes":{"source":{"id":"67763"}},"id":"67765","type":"CDSView"},{"attributes":{},"id":"67733","type":"LinearScale"},{"attributes":{},"id":"67740","type":"BasicTicker"},{"attributes":{},"id":"67729","type":"DataRange1d"},{"attributes":{},"id":"67771","type":"BasicTickFormatter"},{"attributes":{"text":"Centered eight - Non centered eight"},"id":"67766","type":"Title"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"67752","type":"PolyAnnotation"},{"attributes":{"formatter":{"id":"67769"},"ticker":{"id":"67736"}},"id":"67735","type":"LinearAxis"},{"attributes":{"children":[[{"id":"67726"},0,0]]},"id":"67778","type":"GridBox"},{"attributes":{},"id":"67748","type":"UndoTool"},{"attributes":{"axis_label":"ELPD difference","formatter":{"id":"67771"},"ticker":{"id":"67740"}},"id":"67739","type":"LinearAxis"},{"attributes":{},"id":"67727","type":"DataRange1d"},{"attributes":{},"id":"67736","type":"BasicTicker"},{"attributes":{},"id":"67744","type":"PanTool"}],"root_ids":["67781"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"6bab5c6d-77b0-4434-95f4-f4e714e757c6","root_ids":["67781"],"roots":{"67781":"e1f639f2-dc86-4993-b4f1-3f2c0dddd30c"}}];
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