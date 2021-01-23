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
    
      
      
    
      var element = document.getElementById("9bf1f3f9-bb09-4ffa-8009-0484d89169c4");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '9bf1f3f9-bb09-4ffa-8009-0484d89169c4' but no matching script tag was found.")
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
                    
                  var docs_json = '{"7b753bce-ba3a-40bb-9dd6-db739eb6ccfb":{"roots":{"references":[{"attributes":{"data_source":{"id":"3770"},"glyph":{"id":"3771"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"3772"},"selection_glyph":null,"view":{"id":"3774"}},"id":"3773","type":"GlyphRenderer"},{"attributes":{},"id":"3760","type":"ResetTool"},{"attributes":{"formatter":{"id":"3801"},"ticker":{"id":"3753"}},"id":"3752","type":"LinearAxis"},{"attributes":{"text":""},"id":"3775","type":"Title"},{"attributes":{"formatter":{"id":"3803"},"ticker":{"id":"3749"}},"id":"3748","type":"LinearAxis"},{"attributes":{"bottom":{"value":0},"fill_color":{"value":"#000000"},"left":{"field":"left"},"line_alpha":{"value":0},"line_color":{"value":"#000000"},"right":{"field":"right"},"top":{"field":"top"}},"id":"3771","type":"Quad"},{"attributes":{},"id":"3746","type":"LinearScale"},{"attributes":{},"id":"3757","type":"WheelZoomTool"},{"attributes":{},"id":"3759","type":"SaveTool"},{"attributes":{},"id":"3801","type":"BasicTickFormatter"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"3756"},{"id":"3757"},{"id":"3758"},{"id":"3759"},{"id":"3760"},{"id":"3761"}]},"id":"3763","type":"Toolbar"},{"attributes":{},"id":"3749","type":"BasicTicker"},{"attributes":{"axis":{"id":"3748"},"ticker":null},"id":"3751","type":"Grid"},{"attributes":{},"id":"3809","type":"UnionRenderers"},{"attributes":{"overlay":{"id":"3762"}},"id":"3758","type":"BoxZoomTool"},{"attributes":{},"id":"3756","type":"PanTool"},{"attributes":{},"id":"3808","type":"Selection"},{"attributes":{},"id":"3709","type":"DataRange1d"},{"attributes":{"below":[{"id":"3748"}],"center":[{"id":"3751"},{"id":"3755"}],"left":[{"id":"3752"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"3789"}],"title":{"id":"3794"},"toolbar":{"id":"3763"},"x_range":{"id":"3740"},"x_scale":{"id":"3744"},"y_range":{"id":"3742"},"y_scale":{"id":"3746"}},"id":"3739","subtype":"Figure","type":"Plot"},{"attributes":{"data":{"x":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[512]},"y":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[512]}},"selected":{"id":"3808"},"selection_policy":{"id":"3809"}},"id":"3786","type":"ColumnDataSource"},{"attributes":{},"id":"3761","type":"HelpTool"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"3762","type":"BoxAnnotation"},{"attributes":{},"id":"3753","type":"BasicTicker"},{"attributes":{"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"3787","type":"Line"},{"attributes":{"axis":{"id":"3752"},"dimension":1,"ticker":null},"id":"3755","type":"Grid"},{"attributes":{"below":[{"id":"3717"}],"center":[{"id":"3720"},{"id":"3724"},{"id":"3784"}],"left":[{"id":"3721"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"3773"}],"title":{"id":"3775"},"toolbar":{"id":"3732"},"x_range":{"id":"3709"},"x_scale":{"id":"3713"},"y_range":{"id":"3711"},"y_scale":{"id":"3715"}},"id":"3708","subtype":"Figure","type":"Plot"},{"attributes":{"children":[{"id":"3708"},{"id":"3739"}]},"id":"3791","type":"Row"},{"attributes":{},"id":"3803","type":"BasicTickFormatter"},{"attributes":{"data_source":{"id":"3786"},"glyph":{"id":"3787"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"3788"},"selection_glyph":null,"view":{"id":"3790"}},"id":"3789","type":"GlyphRenderer"},{"attributes":{},"id":"3711","type":"DataRange1d"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"3731","type":"BoxAnnotation"},{"attributes":{"source":{"id":"3770"}},"id":"3774","type":"CDSView"},{"attributes":{},"id":"3742","type":"DataRange1d"},{"attributes":{"bottom":{"value":0},"fill_alpha":{"value":0.1},"fill_color":{"value":"#000000"},"left":{"field":"left"},"line_alpha":{"value":0.1},"line_color":{"value":"#000000"},"right":{"field":"right"},"top":{"field":"top"}},"id":"3772","type":"Quad"},{"attributes":{"line_alpha":0.1,"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"3788","type":"Line"},{"attributes":{"formatter":{"id":"3780"},"ticker":{"id":"3718"}},"id":"3717","type":"LinearAxis"},{"attributes":{"formatter":{"id":"3778"},"ticker":{"id":"3722"}},"id":"3721","type":"LinearAxis"},{"attributes":{"source":{"id":"3786"}},"id":"3790","type":"CDSView"},{"attributes":{},"id":"3715","type":"LinearScale"},{"attributes":{},"id":"3713","type":"LinearScale"},{"attributes":{"axis":{"id":"3717"},"ticker":null},"id":"3720","type":"Grid"},{"attributes":{"text":""},"id":"3794","type":"Title"},{"attributes":{},"id":"3730","type":"HelpTool"},{"attributes":{},"id":"3718","type":"BasicTicker"},{"attributes":{},"id":"3783","type":"UnionRenderers"},{"attributes":{"axis":{"id":"3721"},"dimension":1,"ticker":null},"id":"3724","type":"Grid"},{"attributes":{"data":{"left":[0,1,2,3,4,5,6,7,8,9,10,11,12],"right":[1,2,3,4,5,6,7,8,9,10,11,12,13],"top":{"__ndarray__":"eekmMQisnD/8qfHSTWKwPzvfT42XbsI/AiuHFtnOxz9GtvP91HjJP4PAyqFFtsM/IbByaJHtvD8pXI/C9SisP/yp8dJNYqA/nMQgsHJokT/8qfHSTWJwP/yp8dJNYmA//Knx0k1iYD8=","dtype":"float64","order":"little","shape":[13]}},"selected":{"id":"3782"},"selection_policy":{"id":"3783"}},"id":"3770","type":"ColumnDataSource"},{"attributes":{},"id":"3780","type":"BasicTickFormatter"},{"attributes":{},"id":"3728","type":"SaveTool"},{"attributes":{},"id":"3782","type":"Selection"},{"attributes":{},"id":"3722","type":"BasicTicker"},{"attributes":{},"id":"3726","type":"WheelZoomTool"},{"attributes":{"items":[{"id":"3785"}]},"id":"3784","type":"Legend"},{"attributes":{},"id":"3725","type":"PanTool"},{"attributes":{"label":{"value":"Poisson"},"renderers":[{"id":"3773"}]},"id":"3785","type":"LegendItem"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"3725"},{"id":"3726"},{"id":"3727"},{"id":"3728"},{"id":"3729"},{"id":"3730"}]},"id":"3732","type":"Toolbar"},{"attributes":{"overlay":{"id":"3731"}},"id":"3727","type":"BoxZoomTool"},{"attributes":{},"id":"3740","type":"DataRange1d"},{"attributes":{},"id":"3744","type":"LinearScale"},{"attributes":{},"id":"3778","type":"BasicTickFormatter"},{"attributes":{},"id":"3729","type":"ResetTool"}],"root_ids":["3791"]},"title":"Bokeh Application","version":"2.2.3"}}';
                  var render_items = [{"docid":"7b753bce-ba3a-40bb-9dd6-db739eb6ccfb","root_ids":["3791"],"roots":{"3791":"9bf1f3f9-bb09-4ffa-8009-0484d89169c4"}}];
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