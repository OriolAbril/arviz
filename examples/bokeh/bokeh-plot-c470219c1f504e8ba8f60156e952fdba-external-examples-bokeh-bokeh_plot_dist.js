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
    
      
      
    
      var element = document.getElementById("b687d1b6-319a-41f4-95c7-0d87c68ed245");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid 'b687d1b6-319a-41f4-95c7-0d87c68ed245' but no matching script tag was found.")
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
                    
                  var docs_json = '{"cce7f5c7-fcb8-401d-8ea3-cc42c3e4e5b5":{"roots":{"references":[{"attributes":{},"id":"3761","type":"HelpTool"},{"attributes":{},"id":"3749","type":"BasicTicker"},{"attributes":{"axis":{"id":"3748"},"ticker":null},"id":"3751","type":"Grid"},{"attributes":{"axis":{"id":"3752"},"dimension":1,"ticker":null},"id":"3755","type":"Grid"},{"attributes":{"data":{"x":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[512]},"y":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[512]}},"selected":{"id":"3807"},"selection_policy":{"id":"3808"}},"id":"3786","type":"ColumnDataSource"},{"attributes":{},"id":"3728","type":"SaveTool"},{"attributes":{},"id":"3753","type":"BasicTicker"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"3756"},{"id":"3757"},{"id":"3758"},{"id":"3759"},{"id":"3760"},{"id":"3761"}]},"id":"3763","type":"Toolbar"},{"attributes":{"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"3787","type":"Line"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"3731","type":"BoxAnnotation"},{"attributes":{"overlay":{"id":"3731"}},"id":"3727","type":"BoxZoomTool"},{"attributes":{},"id":"3757","type":"WheelZoomTool"},{"attributes":{},"id":"3777","type":"BasicTickFormatter"},{"attributes":{"source":{"id":"3786"}},"id":"3790","type":"CDSView"},{"attributes":{},"id":"3756","type":"PanTool"},{"attributes":{"children":[{"id":"3708"},{"id":"3739"}]},"id":"3791","type":"Row"},{"attributes":{"overlay":{"id":"3762"}},"id":"3758","type":"BoxZoomTool"},{"attributes":{"data_source":{"id":"3786"},"glyph":{"id":"3787"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"3788"},"selection_glyph":null,"view":{"id":"3790"}},"id":"3789","type":"GlyphRenderer"},{"attributes":{},"id":"3759","type":"SaveTool"},{"attributes":{},"id":"3760","type":"ResetTool"},{"attributes":{"bottom":{"value":0},"fill_alpha":{"value":0.1},"fill_color":{"value":"#000000"},"left":{"field":"left"},"line_alpha":{"value":0.1},"line_color":{"value":"#000000"},"right":{"field":"right"},"top":{"field":"top"}},"id":"3772","type":"Quad"},{"attributes":{"line_alpha":0.1,"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"3788","type":"Line"},{"attributes":{"below":[{"id":"3717"}],"center":[{"id":"3720"},{"id":"3724"},{"id":"3784"}],"left":[{"id":"3721"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"3773"}],"title":{"id":"3775"},"toolbar":{"id":"3732"},"x_range":{"id":"3709"},"x_scale":{"id":"3713"},"y_range":{"id":"3711"},"y_scale":{"id":"3715"}},"id":"3708","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"3807","type":"Selection"},{"attributes":{},"id":"3713","type":"LinearScale"},{"attributes":{},"id":"3715","type":"LinearScale"},{"attributes":{},"id":"3781","type":"Selection"},{"attributes":{"formatter":{"id":"3779"},"ticker":{"id":"3722"}},"id":"3721","type":"LinearAxis"},{"attributes":{"formatter":{"id":"3777"},"ticker":{"id":"3718"}},"id":"3717","type":"LinearAxis"},{"attributes":{},"id":"3782","type":"UnionRenderers"},{"attributes":{},"id":"3711","type":"DataRange1d"},{"attributes":{"axis":{"id":"3717"},"ticker":null},"id":"3720","type":"Grid"},{"attributes":{"text":""},"id":"3794","type":"Title"},{"attributes":{},"id":"3808","type":"UnionRenderers"},{"attributes":{"text":""},"id":"3775","type":"Title"},{"attributes":{},"id":"3730","type":"HelpTool"},{"attributes":{"source":{"id":"3770"}},"id":"3774","type":"CDSView"},{"attributes":{},"id":"3718","type":"BasicTicker"},{"attributes":{"data_source":{"id":"3770"},"glyph":{"id":"3771"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"3772"},"selection_glyph":null,"view":{"id":"3774"}},"id":"3773","type":"GlyphRenderer"},{"attributes":{"bottom":{"value":0},"fill_color":{"value":"#000000"},"left":{"field":"left"},"line_alpha":{"value":0},"line_color":{"value":"#000000"},"right":{"field":"right"},"top":{"field":"top"}},"id":"3771","type":"Quad"},{"attributes":{"items":[{"id":"3785"}]},"id":"3784","type":"Legend"},{"attributes":{"label":{"value":"Poisson"},"renderers":[{"id":"3773"}]},"id":"3785","type":"LegendItem"},{"attributes":{"axis":{"id":"3721"},"dimension":1,"ticker":null},"id":"3724","type":"Grid"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"3762","type":"BoxAnnotation"},{"attributes":{},"id":"3722","type":"BasicTicker"},{"attributes":{},"id":"3726","type":"WheelZoomTool"},{"attributes":{},"id":"3725","type":"PanTool"},{"attributes":{"below":[{"id":"3748"}],"center":[{"id":"3751"},{"id":"3755"}],"left":[{"id":"3752"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"3789"}],"title":{"id":"3794"},"toolbar":{"id":"3763"},"x_range":{"id":"3740"},"x_scale":{"id":"3744"},"y_range":{"id":"3742"},"y_scale":{"id":"3746"}},"id":"3739","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"3709","type":"DataRange1d"},{"attributes":{},"id":"3740","type":"DataRange1d"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"3725"},{"id":"3726"},{"id":"3727"},{"id":"3728"},{"id":"3729"},{"id":"3730"}]},"id":"3732","type":"Toolbar"},{"attributes":{},"id":"3729","type":"ResetTool"},{"attributes":{},"id":"3744","type":"LinearScale"},{"attributes":{"formatter":{"id":"3800"},"ticker":{"id":"3749"}},"id":"3748","type":"LinearAxis"},{"attributes":{},"id":"3779","type":"BasicTickFormatter"},{"attributes":{},"id":"3800","type":"BasicTickFormatter"},{"attributes":{},"id":"3742","type":"DataRange1d"},{"attributes":{"data":{"left":[0,1,2,3,4,5,6,7,8,9,10,11],"right":[1,2,3,4,5,6,7,8,9,10,11,12],"top":{"__ndarray__":"Gy/dJAaBlT/TTWIQWDm0P+xRuB6F68E/sp3vp8ZLxz/+1HjpJjHIP9NNYhBYOcQ/eekmMQisvD+cxCCwcmixP0w3iUFg5aA//Knx0k1igD956SYxCKx8P/p+arx0k2g/","dtype":"float64","order":"little","shape":[12]}},"selected":{"id":"3781"},"selection_policy":{"id":"3782"}},"id":"3770","type":"ColumnDataSource"},{"attributes":{"formatter":{"id":"3802"},"ticker":{"id":"3753"}},"id":"3752","type":"LinearAxis"},{"attributes":{},"id":"3802","type":"BasicTickFormatter"},{"attributes":{},"id":"3746","type":"LinearScale"}],"root_ids":["3791"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"cce7f5c7-fcb8-401d-8ea3-cc42c3e4e5b5","root_ids":["3791"],"roots":{"3791":"b687d1b6-319a-41f4-95c7-0d87c68ed245"}}];
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