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
    
      
      
    
      var element = document.getElementById("6a9a7564-3552-437a-ad6a-991eff533396");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '6a9a7564-3552-437a-ad6a-991eff533396' but no matching script tag was found.")
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
                    
                  var docs_json = '{"5d0ee1e5-b9f8-4411-9df9-cb276ce79367":{"roots":{"references":[{"attributes":{},"id":"3713","type":"LinearScale"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"3762","type":"BoxAnnotation"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"3725"},{"id":"3726"},{"id":"3727"},{"id":"3728"},{"id":"3729"},{"id":"3730"}]},"id":"3732","type":"Toolbar"},{"attributes":{"below":[{"id":"3717"}],"center":[{"id":"3720"},{"id":"3724"},{"id":"3784"}],"left":[{"id":"3721"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"3773"}],"title":{"id":"3776"},"toolbar":{"id":"3732"},"x_range":{"id":"3709"},"x_scale":{"id":"3713"},"y_range":{"id":"3711"},"y_scale":{"id":"3715"}},"id":"3708","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"3780","type":"BasicTickFormatter"},{"attributes":{},"id":"3725","type":"PanTool"},{"attributes":{"label":{"value":"Poisson"},"renderers":[{"id":"3773"}]},"id":"3785","type":"LegendItem"},{"attributes":{},"id":"3782","type":"UnionRenderers"},{"attributes":{"formatter":{"id":"3778"},"ticker":{"id":"3722"}},"id":"3721","type":"LinearAxis"},{"attributes":{"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"3787","type":"Line"},{"attributes":{},"id":"3778","type":"BasicTickFormatter"},{"attributes":{"source":{"id":"3786"}},"id":"3790","type":"CDSView"},{"attributes":{},"id":"3801","type":"BasicTickFormatter"},{"attributes":{"children":[{"id":"3708"},{"id":"3739"}]},"id":"3791","type":"Row"},{"attributes":{"below":[{"id":"3748"}],"center":[{"id":"3751"},{"id":"3755"}],"left":[{"id":"3752"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"3789"}],"title":{"id":"3795"},"toolbar":{"id":"3763"},"x_range":{"id":"3740"},"x_scale":{"id":"3744"},"y_range":{"id":"3742"},"y_scale":{"id":"3746"}},"id":"3739","subtype":"Figure","type":"Plot"},{"attributes":{"data_source":{"id":"3786"},"glyph":{"id":"3787"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"3788"},"selection_glyph":null,"view":{"id":"3790"}},"id":"3789","type":"GlyphRenderer"},{"attributes":{},"id":"3729","type":"ResetTool"},{"attributes":{},"id":"3722","type":"BasicTicker"},{"attributes":{},"id":"3783","type":"Selection"},{"attributes":{},"id":"3803","type":"BasicTickFormatter"},{"attributes":{},"id":"3730","type":"HelpTool"},{"attributes":{"formatter":{"id":"3780"},"ticker":{"id":"3718"}},"id":"3717","type":"LinearAxis"},{"attributes":{"line_alpha":0.1,"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"3788","type":"Line"},{"attributes":{},"id":"3808","type":"UnionRenderers"},{"attributes":{"data_source":{"id":"3770"},"glyph":{"id":"3771"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"3772"},"selection_glyph":null,"view":{"id":"3774"}},"id":"3773","type":"GlyphRenderer"},{"attributes":{"data":{"x":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[512]},"y":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[512]}},"selected":{"id":"3809"},"selection_policy":{"id":"3808"}},"id":"3786","type":"ColumnDataSource"},{"attributes":{},"id":"3728","type":"SaveTool"},{"attributes":{"bottom":{"value":0},"fill_alpha":{"value":0.1},"fill_color":{"value":"#000000"},"left":{"field":"left"},"line_alpha":{"value":0.1},"line_color":{"value":"#000000"},"right":{"field":"right"},"top":{"field":"top"}},"id":"3772","type":"Quad"},{"attributes":{"overlay":{"id":"3731"}},"id":"3727","type":"BoxZoomTool"},{"attributes":{},"id":"3718","type":"BasicTicker"},{"attributes":{"items":[{"id":"3785"}]},"id":"3784","type":"Legend"},{"attributes":{"overlay":{"id":"3762"}},"id":"3758","type":"BoxZoomTool"},{"attributes":{},"id":"3760","type":"ResetTool"},{"attributes":{},"id":"3726","type":"WheelZoomTool"},{"attributes":{"text":""},"id":"3795","type":"Title"},{"attributes":{},"id":"3753","type":"BasicTicker"},{"attributes":{"source":{"id":"3770"}},"id":"3774","type":"CDSView"},{"attributes":{},"id":"3744","type":"LinearScale"},{"attributes":{},"id":"3809","type":"Selection"},{"attributes":{"formatter":{"id":"3803"},"ticker":{"id":"3749"}},"id":"3748","type":"LinearAxis"},{"attributes":{},"id":"3742","type":"DataRange1d"},{"attributes":{"bottom":{"value":0},"fill_color":{"value":"#000000"},"left":{"field":"left"},"line_alpha":{"value":0},"line_color":{"value":"#000000"},"right":{"field":"right"},"top":{"field":"top"}},"id":"3771","type":"Quad"},{"attributes":{"data":{"left":[0,1,2,3,4,5,6,7,8,9,10,11],"right":[1,2,3,4,5,6,7,8,9,10,11,12],"top":{"__ndarray__":"O99PjZdukj8bL90kBoG1PycxCKwcWsQ/SgwCK4cWyT/y0k1iEFjJP0SLbOf7qcE/8tJNYhBYuT9Ei2zn+6mxP/p+arx0k5g/O99PjZdugj97FK5H4Xp0P/yp8dJNYlA/","dtype":"float64","order":"little","shape":[12]}},"selected":{"id":"3783"},"selection_policy":{"id":"3782"}},"id":"3770","type":"ColumnDataSource"},{"attributes":{"text":""},"id":"3776","type":"Title"},{"attributes":{"formatter":{"id":"3801"},"ticker":{"id":"3753"}},"id":"3752","type":"LinearAxis"},{"attributes":{},"id":"3757","type":"WheelZoomTool"},{"attributes":{},"id":"3746","type":"LinearScale"},{"attributes":{},"id":"3740","type":"DataRange1d"},{"attributes":{"axis":{"id":"3721"},"dimension":1,"ticker":null},"id":"3724","type":"Grid"},{"attributes":{},"id":"3709","type":"DataRange1d"},{"attributes":{},"id":"3756","type":"PanTool"},{"attributes":{},"id":"3759","type":"SaveTool"},{"attributes":{},"id":"3749","type":"BasicTicker"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"3731","type":"BoxAnnotation"},{"attributes":{"axis":{"id":"3748"},"ticker":null},"id":"3751","type":"Grid"},{"attributes":{},"id":"3761","type":"HelpTool"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"3756"},{"id":"3757"},{"id":"3758"},{"id":"3759"},{"id":"3760"},{"id":"3761"}]},"id":"3763","type":"Toolbar"},{"attributes":{},"id":"3715","type":"LinearScale"},{"attributes":{"axis":{"id":"3752"},"dimension":1,"ticker":null},"id":"3755","type":"Grid"},{"attributes":{},"id":"3711","type":"DataRange1d"},{"attributes":{"axis":{"id":"3717"},"ticker":null},"id":"3720","type":"Grid"}],"root_ids":["3791"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"5d0ee1e5-b9f8-4411-9df9-cb276ce79367","root_ids":["3791"],"roots":{"3791":"6a9a7564-3552-437a-ad6a-991eff533396"}}];
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