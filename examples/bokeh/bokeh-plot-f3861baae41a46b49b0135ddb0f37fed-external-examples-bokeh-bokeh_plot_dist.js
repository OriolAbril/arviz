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
    
      
      
    
      var element = document.getElementById("20efa55a-cc14-4231-8eca-e32b0a07b8c7");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '20efa55a-cc14-4231-8eca-e32b0a07b8c7' but no matching script tag was found.")
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
                    
                  var docs_json = '{"d77994bd-9cdf-4269-937b-cf3044609496":{"roots":{"references":[{"attributes":{},"id":"3742","type":"DataRange1d"},{"attributes":{"data":{"left":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],"right":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],"top":{"__ndarray__":"exSuR+F6hD/0/dR46SaxP4PAyqFFtsM/ukkMAiuHxj/ufD81XrrJP4PAyqFFtsM/0SLb+X5qvD/LoUW28/20P/p+arx0k5g/nMQgsHJokT/8qfHSTWJgP/yp8dJNYlA/AAAAAAAAAAAAAAAAAAAAAPyp8dJNYlA/","dtype":"float64","order":"little","shape":[15]}},"selected":{"id":"3782"},"selection_policy":{"id":"3783"}},"id":"3770","type":"ColumnDataSource"},{"attributes":{"formatter":{"id":"3802"},"ticker":{"id":"3753"}},"id":"3752","type":"LinearAxis"},{"attributes":{},"id":"3800","type":"BasicTickFormatter"},{"attributes":{},"id":"3746","type":"LinearScale"},{"attributes":{},"id":"3761","type":"HelpTool"},{"attributes":{},"id":"3749","type":"BasicTicker"},{"attributes":{},"id":"3777","type":"BasicTickFormatter"},{"attributes":{"axis":{"id":"3748"},"ticker":null},"id":"3751","type":"Grid"},{"attributes":{"axis":{"id":"3752"},"dimension":1,"ticker":null},"id":"3755","type":"Grid"},{"attributes":{"data":{"x":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[512]},"y":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[512]}},"selected":{"id":"3808"},"selection_policy":{"id":"3809"}},"id":"3786","type":"ColumnDataSource"},{"attributes":{},"id":"3728","type":"SaveTool"},{"attributes":{},"id":"3753","type":"BasicTicker"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"3756"},{"id":"3757"},{"id":"3758"},{"id":"3759"},{"id":"3760"},{"id":"3761"}]},"id":"3763","type":"Toolbar"},{"attributes":{"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"3787","type":"Line"},{"attributes":{"overlay":{"id":"3731"}},"id":"3727","type":"BoxZoomTool"},{"attributes":{},"id":"3757","type":"WheelZoomTool"},{"attributes":{},"id":"3756","type":"PanTool"},{"attributes":{"children":[{"id":"3708"},{"id":"3739"}]},"id":"3791","type":"Row"},{"attributes":{"overlay":{"id":"3762"}},"id":"3758","type":"BoxZoomTool"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"3731","type":"BoxAnnotation"},{"attributes":{"data_source":{"id":"3786"},"glyph":{"id":"3787"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"3788"},"selection_glyph":null,"view":{"id":"3790"}},"id":"3789","type":"GlyphRenderer"},{"attributes":{},"id":"3759","type":"SaveTool"},{"attributes":{},"id":"3802","type":"BasicTickFormatter"},{"attributes":{},"id":"3760","type":"ResetTool"},{"attributes":{"bottom":{"value":0},"fill_alpha":{"value":0.1},"fill_color":{"value":"#000000"},"left":{"field":"left"},"line_alpha":{"value":0.1},"line_color":{"value":"#000000"},"right":{"field":"right"},"top":{"field":"top"}},"id":"3772","type":"Quad"},{"attributes":{"line_alpha":0.1,"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"3788","type":"Line"},{"attributes":{},"id":"3779","type":"BasicTickFormatter"},{"attributes":{"source":{"id":"3786"}},"id":"3790","type":"CDSView"},{"attributes":{},"id":"3713","type":"LinearScale"},{"attributes":{},"id":"3715","type":"LinearScale"},{"attributes":{"formatter":{"id":"3779"},"ticker":{"id":"3722"}},"id":"3721","type":"LinearAxis"},{"attributes":{"formatter":{"id":"3777"},"ticker":{"id":"3718"}},"id":"3717","type":"LinearAxis"},{"attributes":{},"id":"3808","type":"Selection"},{"attributes":{},"id":"3711","type":"DataRange1d"},{"attributes":{"axis":{"id":"3717"},"ticker":null},"id":"3720","type":"Grid"},{"attributes":{"text":""},"id":"3795","type":"Title"},{"attributes":{"text":""},"id":"3776","type":"Title"},{"attributes":{},"id":"3730","type":"HelpTool"},{"attributes":{},"id":"3782","type":"Selection"},{"attributes":{},"id":"3718","type":"BasicTicker"},{"attributes":{"source":{"id":"3770"}},"id":"3774","type":"CDSView"},{"attributes":{"data_source":{"id":"3770"},"glyph":{"id":"3771"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"3772"},"selection_glyph":null,"view":{"id":"3774"}},"id":"3773","type":"GlyphRenderer"},{"attributes":{"bottom":{"value":0},"fill_color":{"value":"#000000"},"left":{"field":"left"},"line_alpha":{"value":0},"line_color":{"value":"#000000"},"right":{"field":"right"},"top":{"field":"top"}},"id":"3771","type":"Quad"},{"attributes":{"items":[{"id":"3785"}]},"id":"3784","type":"Legend"},{"attributes":{},"id":"3809","type":"UnionRenderers"},{"attributes":{"label":{"value":"Poisson"},"renderers":[{"id":"3773"}]},"id":"3785","type":"LegendItem"},{"attributes":{"axis":{"id":"3721"},"dimension":1,"ticker":null},"id":"3724","type":"Grid"},{"attributes":{},"id":"3783","type":"UnionRenderers"},{"attributes":{},"id":"3722","type":"BasicTicker"},{"attributes":{},"id":"3726","type":"WheelZoomTool"},{"attributes":{"below":[{"id":"3717"}],"center":[{"id":"3720"},{"id":"3724"},{"id":"3784"}],"left":[{"id":"3721"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"3773"}],"title":{"id":"3776"},"toolbar":{"id":"3732"},"x_range":{"id":"3709"},"x_scale":{"id":"3713"},"y_range":{"id":"3711"},"y_scale":{"id":"3715"}},"id":"3708","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"3725","type":"PanTool"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"3762","type":"BoxAnnotation"},{"attributes":{"below":[{"id":"3748"}],"center":[{"id":"3751"},{"id":"3755"}],"left":[{"id":"3752"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"3789"}],"title":{"id":"3795"},"toolbar":{"id":"3763"},"x_range":{"id":"3740"},"x_scale":{"id":"3744"},"y_range":{"id":"3742"},"y_scale":{"id":"3746"}},"id":"3739","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"3709","type":"DataRange1d"},{"attributes":{},"id":"3740","type":"DataRange1d"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"3725"},{"id":"3726"},{"id":"3727"},{"id":"3728"},{"id":"3729"},{"id":"3730"}]},"id":"3732","type":"Toolbar"},{"attributes":{},"id":"3729","type":"ResetTool"},{"attributes":{},"id":"3744","type":"LinearScale"},{"attributes":{"formatter":{"id":"3800"},"ticker":{"id":"3749"}},"id":"3748","type":"LinearAxis"}],"root_ids":["3791"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"d77994bd-9cdf-4269-937b-cf3044609496","root_ids":["3791"],"roots":{"3791":"20efa55a-cc14-4231-8eca-e32b0a07b8c7"}}];
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