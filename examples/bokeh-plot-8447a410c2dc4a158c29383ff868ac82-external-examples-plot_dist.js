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
    
      
      
    
      var element = document.getElementById("51594903-f812-4436-81c0-99435534c93d");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '51594903-f812-4436-81c0-99435534c93d' but no matching script tag was found.")
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
                    
                  var docs_json = '{"5f60eb8e-ac0d-4d8e-9798-301714693673":{"roots":{"references":[{"attributes":{},"id":"3749","type":"BasicTicker"},{"attributes":{"bottom":{"value":0},"fill_color":{"value":"#000000"},"left":{"field":"left"},"line_alpha":{"value":0},"line_color":{"value":"#000000"},"right":{"field":"right"},"top":{"field":"top"}},"id":"3771","type":"Quad"},{"attributes":{},"id":"3757","type":"WheelZoomTool"},{"attributes":{"formatter":{"id":"3800"},"ticker":{"id":"3749"}},"id":"3748","type":"LinearAxis"},{"attributes":{},"id":"3800","type":"BasicTickFormatter"},{"attributes":{},"id":"3759","type":"SaveTool"},{"attributes":{"formatter":{"id":"3802"},"ticker":{"id":"3753"}},"id":"3752","type":"LinearAxis"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"3756"},{"id":"3757"},{"id":"3758"},{"id":"3759"},{"id":"3760"},{"id":"3761"}]},"id":"3763","type":"Toolbar"},{"attributes":{"axis":{"id":"3748"},"ticker":null},"id":"3751","type":"Grid"},{"attributes":{"overlay":{"id":"3762"}},"id":"3758","type":"BoxZoomTool"},{"attributes":{},"id":"3777","type":"BasicTickFormatter"},{"attributes":{},"id":"3756","type":"PanTool"},{"attributes":{},"id":"3709","type":"DataRange1d"},{"attributes":{"below":[{"id":"3748"}],"center":[{"id":"3751"},{"id":"3755"}],"left":[{"id":"3752"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"3789"}],"title":{"id":"3794"},"toolbar":{"id":"3763"},"x_range":{"id":"3740"},"x_scale":{"id":"3744"},"y_range":{"id":"3742"},"y_scale":{"id":"3746"}},"id":"3739","subtype":"Figure","type":"Plot"},{"attributes":{"data":{"x":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[512]},"y":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[512]}},"selected":{"id":"3807"},"selection_policy":{"id":"3808"}},"id":"3786","type":"ColumnDataSource"},{"attributes":{"below":[{"id":"3717"}],"center":[{"id":"3720"},{"id":"3724"},{"id":"3784"}],"left":[{"id":"3721"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"3773"}],"title":{"id":"3775"},"toolbar":{"id":"3732"},"x_range":{"id":"3709"},"x_scale":{"id":"3713"},"y_range":{"id":"3711"},"y_scale":{"id":"3715"}},"id":"3708","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"3761","type":"HelpTool"},{"attributes":{},"id":"3753","type":"BasicTicker"},{"attributes":{"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"3787","type":"Line"},{"attributes":{"axis":{"id":"3752"},"dimension":1,"ticker":null},"id":"3755","type":"Grid"},{"attributes":{"source":{"id":"3786"}},"id":"3790","type":"CDSView"},{"attributes":{"children":[{"id":"3708"},{"id":"3739"}]},"id":"3791","type":"Row"},{"attributes":{},"id":"3802","type":"BasicTickFormatter"},{"attributes":{"data_source":{"id":"3786"},"glyph":{"id":"3787"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"3788"},"selection_glyph":null,"view":{"id":"3790"}},"id":"3789","type":"GlyphRenderer"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"3731","type":"BoxAnnotation"},{"attributes":{},"id":"3711","type":"DataRange1d"},{"attributes":{},"id":"3744","type":"LinearScale"},{"attributes":{"line_alpha":0.1,"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"3788","type":"Line"},{"attributes":{"formatter":{"id":"3777"},"ticker":{"id":"3718"}},"id":"3717","type":"LinearAxis"},{"attributes":{},"id":"3779","type":"BasicTickFormatter"},{"attributes":{},"id":"3807","type":"Selection"},{"attributes":{"formatter":{"id":"3779"},"ticker":{"id":"3722"}},"id":"3721","type":"LinearAxis"},{"attributes":{},"id":"3715","type":"LinearScale"},{"attributes":{},"id":"3713","type":"LinearScale"},{"attributes":{},"id":"3781","type":"Selection"},{"attributes":{"axis":{"id":"3717"},"ticker":null},"id":"3720","type":"Grid"},{"attributes":{"data_source":{"id":"3770"},"glyph":{"id":"3771"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"3772"},"selection_glyph":null,"view":{"id":"3774"}},"id":"3773","type":"GlyphRenderer"},{"attributes":{},"id":"3782","type":"UnionRenderers"},{"attributes":{"label":{"value":"Poisson"},"renderers":[{"id":"3773"}]},"id":"3785","type":"LegendItem"},{"attributes":{},"id":"3730","type":"HelpTool"},{"attributes":{},"id":"3808","type":"UnionRenderers"},{"attributes":{"bottom":{"value":0},"fill_alpha":{"value":0.1},"fill_color":{"value":"#000000"},"left":{"field":"left"},"line_alpha":{"value":0.1},"line_color":{"value":"#000000"},"right":{"field":"right"},"top":{"field":"top"}},"id":"3772","type":"Quad"},{"attributes":{},"id":"3718","type":"BasicTicker"},{"attributes":{"items":[{"id":"3785"}]},"id":"3784","type":"Legend"},{"attributes":{"text":""},"id":"3794","type":"Title"},{"attributes":{"source":{"id":"3770"}},"id":"3774","type":"CDSView"},{"attributes":{"axis":{"id":"3721"},"dimension":1,"ticker":null},"id":"3724","type":"Grid"},{"attributes":{"data":{"left":[0,1,2,3,4,5,6,7,8,9,10,11,12],"right":[1,2,3,4,5,6,7,8,9,10,11,12,13],"top":{"__ndarray__":"mpmZmZmZmT/sUbgeheuxPzMzMzMzM8M/SgwCK4cWyT+e76fGSzfJPz81XrpJDMI/SgwCK4cWuT/8qfHSTWKwP9v5fmq8dKM/+n5qvHSTiD97FK5H4Xp0P/p+arx0k2g//Knx0k1iUD8=","dtype":"float64","order":"little","shape":[13]}},"selected":{"id":"3781"},"selection_policy":{"id":"3782"}},"id":"3770","type":"ColumnDataSource"},{"attributes":{},"id":"3728","type":"SaveTool"},{"attributes":{},"id":"3722","type":"BasicTicker"},{"attributes":{"text":""},"id":"3775","type":"Title"},{"attributes":{},"id":"3726","type":"WheelZoomTool"},{"attributes":{},"id":"3725","type":"PanTool"},{"attributes":{"overlay":{"id":"3731"}},"id":"3727","type":"BoxZoomTool"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"3725"},{"id":"3726"},{"id":"3727"},{"id":"3728"},{"id":"3729"},{"id":"3730"}]},"id":"3732","type":"Toolbar"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"3762","type":"BoxAnnotation"},{"attributes":{},"id":"3746","type":"LinearScale"},{"attributes":{},"id":"3740","type":"DataRange1d"},{"attributes":{},"id":"3742","type":"DataRange1d"},{"attributes":{},"id":"3729","type":"ResetTool"},{"attributes":{},"id":"3760","type":"ResetTool"}],"root_ids":["3791"]},"title":"Bokeh Application","version":"2.2.3"}}';
                  var render_items = [{"docid":"5f60eb8e-ac0d-4d8e-9798-301714693673","root_ids":["3791"],"roots":{"3791":"51594903-f812-4436-81c0-99435534c93d"}}];
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