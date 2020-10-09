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
    
      
      
    
      var element = document.getElementById("18a60614-1d56-4afa-a02e-eec7030d62ae");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '18a60614-1d56-4afa-a02e-eec7030d62ae' but no matching script tag was found.")
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
                    
                  var docs_json = '{"db3a6be0-13af-4710-98df-dbde4808f2ea":{"roots":{"references":[{"attributes":{"axis":{"id":"3752"},"dimension":1,"ticker":null},"id":"3755","type":"Grid"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"3731","type":"BoxAnnotation"},{"attributes":{},"id":"3722","type":"BasicTicker"},{"attributes":{},"id":"3726","type":"WheelZoomTool"},{"attributes":{},"id":"3711","type":"DataRange1d"},{"attributes":{"below":[{"id":"3717"}],"center":[{"id":"3720"},{"id":"3724"},{"id":"3784"}],"left":[{"id":"3721"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"3773"}],"title":{"id":"3775"},"toolbar":{"id":"3732"},"x_range":{"id":"3709"},"x_scale":{"id":"3713"},"y_range":{"id":"3711"},"y_scale":{"id":"3715"}},"id":"3708","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"3782","type":"UnionRenderers"},{"attributes":{"label":{"value":"Poisson"},"renderers":[{"id":"3773"}]},"id":"3785","type":"LegendItem"},{"attributes":{"formatter":{"id":"3777"},"ticker":{"id":"3722"}},"id":"3721","type":"LinearAxis"},{"attributes":{},"id":"3779","type":"BasicTickFormatter"},{"attributes":{"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"3787","type":"Line"},{"attributes":{"source":{"id":"3786"}},"id":"3790","type":"CDSView"},{"attributes":{"children":[{"id":"3708"},{"id":"3739"}]},"id":"3791","type":"Row"},{"attributes":{},"id":"3802","type":"BasicTickFormatter"},{"attributes":{"data_source":{"id":"3786"},"glyph":{"id":"3787"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"3788"},"selection_glyph":null,"view":{"id":"3790"}},"id":"3789","type":"GlyphRenderer"},{"attributes":{},"id":"3729","type":"ResetTool"},{"attributes":{"bottom":{"value":0},"fill_color":{"value":"#000000"},"left":{"field":"left"},"line_alpha":{"value":0},"line_color":{"value":"#000000"},"right":{"field":"right"},"top":{"field":"top"}},"id":"3771","type":"Quad"},{"attributes":{"line_alpha":0.1,"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"3788","type":"Line"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"3725"},{"id":"3726"},{"id":"3727"},{"id":"3728"},{"id":"3729"},{"id":"3730"}]},"id":"3732","type":"Toolbar"},{"attributes":{"below":[{"id":"3748"}],"center":[{"id":"3751"},{"id":"3755"}],"left":[{"id":"3752"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"3789"}],"title":{"id":"3794"},"toolbar":{"id":"3763"},"x_range":{"id":"3740"},"x_scale":{"id":"3744"},"y_range":{"id":"3742"},"y_scale":{"id":"3746"}},"id":"3739","subtype":"Figure","type":"Plot"},{"attributes":{"formatter":{"id":"3779"},"ticker":{"id":"3718"}},"id":"3717","type":"LinearAxis"},{"attributes":{},"id":"3718","type":"BasicTicker"},{"attributes":{},"id":"3730","type":"HelpTool"},{"attributes":{"data":{"x":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[512]},"y":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[512]}},"selected":{"id":"3807"},"selection_policy":{"id":"3808"}},"id":"3786","type":"ColumnDataSource"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"3762","type":"BoxAnnotation"},{"attributes":{"bottom":{"value":0},"fill_alpha":{"value":0.1},"fill_color":{"value":"#000000"},"left":{"field":"left"},"line_alpha":{"value":0.1},"line_color":{"value":"#000000"},"right":{"field":"right"},"top":{"field":"top"}},"id":"3772","type":"Quad"},{"attributes":{},"id":"3757","type":"WheelZoomTool"},{"attributes":{"data_source":{"id":"3770"},"glyph":{"id":"3771"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"3772"},"selection_glyph":null,"view":{"id":"3774"}},"id":"3773","type":"GlyphRenderer"},{"attributes":{},"id":"3709","type":"DataRange1d"},{"attributes":{},"id":"3807","type":"Selection"},{"attributes":{},"id":"3744","type":"LinearScale"},{"attributes":{"text":""},"id":"3794","type":"Title"},{"attributes":{"source":{"id":"3770"}},"id":"3774","type":"CDSView"},{"attributes":{"formatter":{"id":"3802"},"ticker":{"id":"3749"}},"id":"3748","type":"LinearAxis"},{"attributes":{"overlay":{"id":"3731"}},"id":"3727","type":"BoxZoomTool"},{"attributes":{},"id":"3808","type":"UnionRenderers"},{"attributes":{},"id":"3742","type":"DataRange1d"},{"attributes":{"items":[{"id":"3785"}]},"id":"3784","type":"Legend"},{"attributes":{"data":{"left":[0,1,2,3,4,5,6,7,8,9,10,11,12],"right":[1,2,3,4,5,6,7,8,9,10,11,12,13],"top":{"__ndarray__":"O99PjZdukj+6SQwCK4e2P3sUrkfhesQ/ke18PzVeyj9qvHSTGATGP3e+nxov3cQ/+n5qvHSTuD+4HoXrUbiuP5qZmZmZmZk/exSuR+F6dD97FK5H4Xp0PwAAAAAAAAAA/Knx0k1iYD8=","dtype":"float64","order":"little","shape":[13]}},"selected":{"id":"3781"},"selection_policy":{"id":"3782"}},"id":"3770","type":"ColumnDataSource"},{"attributes":{},"id":"3760","type":"ResetTool"},{"attributes":{"formatter":{"id":"3800"},"ticker":{"id":"3753"}},"id":"3752","type":"LinearAxis"},{"attributes":{},"id":"3715","type":"LinearScale"},{"attributes":{},"id":"3746","type":"LinearScale"},{"attributes":{},"id":"3756","type":"PanTool"},{"attributes":{},"id":"3728","type":"SaveTool"},{"attributes":{},"id":"3800","type":"BasicTickFormatter"},{"attributes":{"overlay":{"id":"3762"}},"id":"3758","type":"BoxZoomTool"},{"attributes":{"text":""},"id":"3775","type":"Title"},{"attributes":{},"id":"3759","type":"SaveTool"},{"attributes":{},"id":"3749","type":"BasicTicker"},{"attributes":{},"id":"3740","type":"DataRange1d"},{"attributes":{"axis":{"id":"3748"},"ticker":null},"id":"3751","type":"Grid"},{"attributes":{},"id":"3777","type":"BasicTickFormatter"},{"attributes":{},"id":"3781","type":"Selection"},{"attributes":{},"id":"3761","type":"HelpTool"},{"attributes":{"axis":{"id":"3721"},"dimension":1,"ticker":null},"id":"3724","type":"Grid"},{"attributes":{},"id":"3713","type":"LinearScale"},{"attributes":{},"id":"3753","type":"BasicTicker"},{"attributes":{"axis":{"id":"3717"},"ticker":null},"id":"3720","type":"Grid"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"3756"},{"id":"3757"},{"id":"3758"},{"id":"3759"},{"id":"3760"},{"id":"3761"}]},"id":"3763","type":"Toolbar"},{"attributes":{},"id":"3725","type":"PanTool"}],"root_ids":["3791"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"db3a6be0-13af-4710-98df-dbde4808f2ea","root_ids":["3791"],"roots":{"3791":"18a60614-1d56-4afa-a02e-eec7030d62ae"}}];
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