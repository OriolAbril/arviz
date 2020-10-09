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
    
      
      
    
      var element = document.getElementById("7995da46-5814-45c0-9ac4-705b8b8415b0");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '7995da46-5814-45c0-9ac4-705b8b8415b0' but no matching script tag was found.")
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
                    
                  var docs_json = '{"e92f10ea-01fd-44f3-89ec-ee24e666a5cf":{"roots":{"references":[{"attributes":{"formatter":{"id":"3871"},"ticker":{"id":"3810"}},"id":"3809","type":"LinearAxis"},{"attributes":{"formatter":{"id":"3894"},"ticker":{"id":"3841"}},"id":"3840","type":"LinearAxis"},{"attributes":{"source":{"id":"3862"}},"id":"3866","type":"CDSView"},{"attributes":{},"id":"3900","type":"UnionRenderers"},{"attributes":{"items":[{"id":"3877"}]},"id":"3876","type":"Legend"},{"attributes":{},"id":"3901","type":"Selection"},{"attributes":{"data":{"x":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[512]},"y":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[512]}},"selected":{"id":"3901"},"selection_policy":{"id":"3900"}},"id":"3878","type":"ColumnDataSource"},{"attributes":{},"id":"3820","type":"SaveTool"},{"attributes":{},"id":"3875","type":"Selection"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"3854","type":"BoxAnnotation"},{"attributes":{"overlay":{"id":"3823"}},"id":"3819","type":"BoxZoomTool"},{"attributes":{},"id":"3817","type":"PanTool"},{"attributes":{},"id":"3852","type":"ResetTool"},{"attributes":{"axis":{"id":"3813"},"dimension":1,"ticker":null},"id":"3816","type":"Grid"},{"attributes":{"text":""},"id":"3868","type":"Title"},{"attributes":{},"id":"3849","type":"WheelZoomTool"},{"attributes":{},"id":"3874","type":"UnionRenderers"},{"attributes":{},"id":"3807","type":"LinearScale"},{"attributes":{},"id":"3821","type":"ResetTool"},{"attributes":{"bottom":{"value":0},"fill_alpha":{"value":0.1},"fill_color":{"value":"#000000"},"left":{"field":"left"},"line_alpha":{"value":0.1},"line_color":{"value":"#000000"},"right":{"field":"right"},"top":{"field":"top"}},"id":"3864","type":"Quad"},{"attributes":{},"id":"3845","type":"BasicTicker"},{"attributes":{},"id":"3814","type":"BasicTicker"},{"attributes":{},"id":"3871","type":"BasicTickFormatter"},{"attributes":{},"id":"3892","type":"BasicTickFormatter"},{"attributes":{},"id":"3803","type":"DataRange1d"},{"attributes":{},"id":"3818","type":"WheelZoomTool"},{"attributes":{},"id":"3841","type":"BasicTicker"},{"attributes":{"axis":{"id":"3840"},"ticker":null},"id":"3843","type":"Grid"},{"attributes":{},"id":"3838","type":"LinearScale"},{"attributes":{},"id":"3853","type":"HelpTool"},{"attributes":{"data_source":{"id":"3862"},"glyph":{"id":"3863"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"3864"},"selection_glyph":null,"view":{"id":"3866"}},"id":"3865","type":"GlyphRenderer"},{"attributes":{"formatter":{"id":"3892"},"ticker":{"id":"3845"}},"id":"3844","type":"LinearAxis"},{"attributes":{"label":{"value":"Poisson"},"renderers":[{"id":"3865"}]},"id":"3877","type":"LegendItem"},{"attributes":{},"id":"3805","type":"LinearScale"},{"attributes":{},"id":"3801","type":"DataRange1d"},{"attributes":{},"id":"3834","type":"DataRange1d"},{"attributes":{"data":{"left":[0,1,2,3,4,5,6,7,8,9,10,11,12],"right":[1,2,3,4,5,6,7,8,9,10,11,12,13],"top":{"__ndarray__":"O99PjZdukj8rhxbZzvezP+Olm8QgsMI/qvHSTWIQyD8dWmQ730/NP5huEoPAysE/6SYxCKwcuj85tMh2vp+qP7gehetRuJ4/ukkMAiuHhj/8qfHSTWJgP/p+arx0k2g//Knx0k1iYD8=","dtype":"float64","order":"little","shape":[13]}},"selected":{"id":"3875"},"selection_policy":{"id":"3874"}},"id":"3862","type":"ColumnDataSource"},{"attributes":{"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"3879","type":"Line"},{"attributes":{},"id":"3894","type":"BasicTickFormatter"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"3823","type":"BoxAnnotation"},{"attributes":{},"id":"3869","type":"BasicTickFormatter"},{"attributes":{"formatter":{"id":"3869"},"ticker":{"id":"3814"}},"id":"3813","type":"LinearAxis"},{"attributes":{"children":[{"id":"3800"},{"id":"3831"}]},"id":"3883","type":"Row"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"3848"},{"id":"3849"},{"id":"3850"},{"id":"3851"},{"id":"3852"},{"id":"3853"}]},"id":"3855","type":"Toolbar"},{"attributes":{"data_source":{"id":"3878"},"glyph":{"id":"3879"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"3880"},"selection_glyph":null,"view":{"id":"3882"}},"id":"3881","type":"GlyphRenderer"},{"attributes":{},"id":"3822","type":"HelpTool"},{"attributes":{},"id":"3832","type":"DataRange1d"},{"attributes":{"below":[{"id":"3840"}],"center":[{"id":"3843"},{"id":"3847"}],"left":[{"id":"3844"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"3881"}],"title":{"id":"3887"},"toolbar":{"id":"3855"},"x_range":{"id":"3832"},"x_scale":{"id":"3836"},"y_range":{"id":"3834"},"y_scale":{"id":"3838"}},"id":"3831","subtype":"Figure","type":"Plot"},{"attributes":{"line_alpha":0.1,"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"3880","type":"Line"},{"attributes":{"below":[{"id":"3809"}],"center":[{"id":"3812"},{"id":"3816"},{"id":"3876"}],"left":[{"id":"3813"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"3865"}],"title":{"id":"3868"},"toolbar":{"id":"3824"},"x_range":{"id":"3801"},"x_scale":{"id":"3805"},"y_range":{"id":"3803"},"y_scale":{"id":"3807"}},"id":"3800","subtype":"Figure","type":"Plot"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"3817"},{"id":"3818"},{"id":"3819"},{"id":"3820"},{"id":"3821"},{"id":"3822"}]},"id":"3824","type":"Toolbar"},{"attributes":{"overlay":{"id":"3854"}},"id":"3850","type":"BoxZoomTool"},{"attributes":{"source":{"id":"3878"}},"id":"3882","type":"CDSView"},{"attributes":{},"id":"3848","type":"PanTool"},{"attributes":{"axis":{"id":"3809"},"ticker":null},"id":"3812","type":"Grid"},{"attributes":{"bottom":{"value":0},"fill_color":{"value":"#000000"},"left":{"field":"left"},"line_alpha":{"value":0},"line_color":{"value":"#000000"},"right":{"field":"right"},"top":{"field":"top"}},"id":"3863","type":"Quad"},{"attributes":{},"id":"3851","type":"SaveTool"},{"attributes":{},"id":"3810","type":"BasicTicker"},{"attributes":{},"id":"3836","type":"LinearScale"},{"attributes":{"axis":{"id":"3844"},"dimension":1,"ticker":null},"id":"3847","type":"Grid"},{"attributes":{"text":""},"id":"3887","type":"Title"}],"root_ids":["3883"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"e92f10ea-01fd-44f3-89ec-ee24e666a5cf","root_ids":["3883"],"roots":{"3883":"7995da46-5814-45c0-9ac4-705b8b8415b0"}}];
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