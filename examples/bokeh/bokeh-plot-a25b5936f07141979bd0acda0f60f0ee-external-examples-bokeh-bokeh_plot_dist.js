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
    
      
      
    
      var element = document.getElementById("f689b288-6f4c-4978-88da-a45ec1ee51fa");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid 'f689b288-6f4c-4978-88da-a45ec1ee51fa' but no matching script tag was found.")
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
                    
                  var docs_json = '{"1be183e5-32c1-4fdd-b806-1a79a221314e":{"roots":{"references":[{"attributes":{},"id":"3744","type":"LinearScale"},{"attributes":{"formatter":{"id":"3802"},"ticker":{"id":"3749"}},"id":"3748","type":"LinearAxis"},{"attributes":{},"id":"3742","type":"DataRange1d"},{"attributes":{},"id":"3782","type":"Selection"},{"attributes":{"data":{"left":[0,1,2,3,4,5,6,7,8,9,10,11,12,13],"right":[1,2,3,4,5,6,7,8,9,10,11,12,13,14],"top":{"__ndarray__":"eekmMQisjD/b+X5qvHSzP99PjZduEsM/lkOLbOf7yT+WQ4ts5/vJP2q8dJMYBMY/EoPAyqFFtj/Jdr6fGi+tP7pJDAIrh5Y/eekmMQisfD/6fmq8dJN4P/p+arx0k2g/AAAAAAAAAAD8qfHSTWJQPw==","dtype":"float64","order":"little","shape":[14]}},"selected":{"id":"3782"},"selection_policy":{"id":"3783"}},"id":"3770","type":"ColumnDataSource"},{"attributes":{},"id":"3783","type":"UnionRenderers"},{"attributes":{"formatter":{"id":"3800"},"ticker":{"id":"3753"}},"id":"3752","type":"LinearAxis"},{"attributes":{},"id":"3746","type":"LinearScale"},{"attributes":{},"id":"3761","type":"HelpTool"},{"attributes":{},"id":"3800","type":"BasicTickFormatter"},{"attributes":{},"id":"3749","type":"BasicTicker"},{"attributes":{"axis":{"id":"3748"},"ticker":null},"id":"3751","type":"Grid"},{"attributes":{},"id":"3777","type":"BasicTickFormatter"},{"attributes":{},"id":"3779","type":"BasicTickFormatter"},{"attributes":{"axis":{"id":"3752"},"dimension":1,"ticker":null},"id":"3755","type":"Grid"},{"attributes":{"data":{"x":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[512]},"y":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[512]}},"selected":{"id":"3808"},"selection_policy":{"id":"3809"}},"id":"3786","type":"ColumnDataSource"},{"attributes":{},"id":"3802","type":"BasicTickFormatter"},{"attributes":{},"id":"3728","type":"SaveTool"},{"attributes":{},"id":"3753","type":"BasicTicker"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"3756"},{"id":"3757"},{"id":"3758"},{"id":"3759"},{"id":"3760"},{"id":"3761"}]},"id":"3763","type":"Toolbar"},{"attributes":{"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"3787","type":"Line"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"3731","type":"BoxAnnotation"},{"attributes":{"overlay":{"id":"3731"}},"id":"3727","type":"BoxZoomTool"},{"attributes":{},"id":"3757","type":"WheelZoomTool"},{"attributes":{},"id":"3756","type":"PanTool"},{"attributes":{"children":[{"id":"3708"},{"id":"3739"}]},"id":"3791","type":"Row"},{"attributes":{"overlay":{"id":"3762"}},"id":"3758","type":"BoxZoomTool"},{"attributes":{"data_source":{"id":"3786"},"glyph":{"id":"3787"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"3788"},"selection_glyph":null,"view":{"id":"3790"}},"id":"3789","type":"GlyphRenderer"},{"attributes":{},"id":"3759","type":"SaveTool"},{"attributes":{},"id":"3760","type":"ResetTool"},{"attributes":{"below":[{"id":"3717"}],"center":[{"id":"3720"},{"id":"3724"},{"id":"3784"}],"left":[{"id":"3721"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"3773"}],"title":{"id":"3776"},"toolbar":{"id":"3732"},"x_range":{"id":"3709"},"x_scale":{"id":"3713"},"y_range":{"id":"3711"},"y_scale":{"id":"3715"}},"id":"3708","subtype":"Figure","type":"Plot"},{"attributes":{"bottom":{"value":0},"fill_alpha":{"value":0.1},"fill_color":{"value":"#000000"},"left":{"field":"left"},"line_alpha":{"value":0.1},"line_color":{"value":"#000000"},"right":{"field":"right"},"top":{"field":"top"}},"id":"3772","type":"Quad"},{"attributes":{"line_alpha":0.1,"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"3788","type":"Line"},{"attributes":{"source":{"id":"3786"}},"id":"3790","type":"CDSView"},{"attributes":{},"id":"3713","type":"LinearScale"},{"attributes":{},"id":"3715","type":"LinearScale"},{"attributes":{"formatter":{"id":"3777"},"ticker":{"id":"3722"}},"id":"3721","type":"LinearAxis"},{"attributes":{"text":""},"id":"3776","type":"Title"},{"attributes":{"formatter":{"id":"3779"},"ticker":{"id":"3718"}},"id":"3717","type":"LinearAxis"},{"attributes":{},"id":"3711","type":"DataRange1d"},{"attributes":{"axis":{"id":"3717"},"ticker":null},"id":"3720","type":"Grid"},{"attributes":{},"id":"3730","type":"HelpTool"},{"attributes":{},"id":"3718","type":"BasicTicker"},{"attributes":{"source":{"id":"3770"}},"id":"3774","type":"CDSView"},{"attributes":{"data_source":{"id":"3770"},"glyph":{"id":"3771"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"3772"},"selection_glyph":null,"view":{"id":"3774"}},"id":"3773","type":"GlyphRenderer"},{"attributes":{"text":""},"id":"3795","type":"Title"},{"attributes":{"bottom":{"value":0},"fill_color":{"value":"#000000"},"left":{"field":"left"},"line_alpha":{"value":0},"line_color":{"value":"#000000"},"right":{"field":"right"},"top":{"field":"top"}},"id":"3771","type":"Quad"},{"attributes":{"items":[{"id":"3785"}]},"id":"3784","type":"Legend"},{"attributes":{"label":{"value":"Poisson"},"renderers":[{"id":"3773"}]},"id":"3785","type":"LegendItem"},{"attributes":{"axis":{"id":"3721"},"dimension":1,"ticker":null},"id":"3724","type":"Grid"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"3762","type":"BoxAnnotation"},{"attributes":{},"id":"3722","type":"BasicTicker"},{"attributes":{},"id":"3726","type":"WheelZoomTool"},{"attributes":{},"id":"3725","type":"PanTool"},{"attributes":{"below":[{"id":"3748"}],"center":[{"id":"3751"},{"id":"3755"}],"left":[{"id":"3752"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"3789"}],"title":{"id":"3795"},"toolbar":{"id":"3763"},"x_range":{"id":"3740"},"x_scale":{"id":"3744"},"y_range":{"id":"3742"},"y_scale":{"id":"3746"}},"id":"3739","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"3729","type":"ResetTool"},{"attributes":{},"id":"3709","type":"DataRange1d"},{"attributes":{},"id":"3740","type":"DataRange1d"},{"attributes":{},"id":"3808","type":"Selection"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"3725"},{"id":"3726"},{"id":"3727"},{"id":"3728"},{"id":"3729"},{"id":"3730"}]},"id":"3732","type":"Toolbar"},{"attributes":{},"id":"3809","type":"UnionRenderers"}],"root_ids":["3791"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"1be183e5-32c1-4fdd-b806-1a79a221314e","root_ids":["3791"],"roots":{"3791":"f689b288-6f4c-4978-88da-a45ec1ee51fa"}}];
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