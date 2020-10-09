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
    
      
      
    
      var element = document.getElementById("37e23653-e59e-408a-9a5c-8c0e4f495c85");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '37e23653-e59e-408a-9a5c-8c0e4f495c85' but no matching script tag was found.")
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
                    
                  var docs_json = '{"38fc5042-929e-4938-8d94-a56cda94e534":{"roots":{"references":[{"attributes":{},"id":"3899","type":"UnionRenderers"},{"attributes":{},"id":"3900","type":"Selection"},{"attributes":{},"id":"3873","type":"UnionRenderers"},{"attributes":{"data":{"x":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[512]},"y":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[512]}},"selected":{"id":"3900"},"selection_policy":{"id":"3899"}},"id":"3878","type":"ColumnDataSource"},{"attributes":{},"id":"3810","type":"BasicTicker"},{"attributes":{"items":[{"id":"3877"}]},"id":"3876","type":"Legend"},{"attributes":{},"id":"3892","type":"BasicTickFormatter"},{"attributes":{"formatter":{"id":"3871"},"ticker":{"id":"3810"}},"id":"3809","type":"LinearAxis"},{"attributes":{},"id":"3817","type":"PanTool"},{"attributes":{"text":""},"id":"3867","type":"Title"},{"attributes":{"below":[{"id":"3840"}],"center":[{"id":"3843"},{"id":"3847"}],"left":[{"id":"3844"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"3881"}],"title":{"id":"3886"},"toolbar":{"id":"3855"},"x_range":{"id":"3832"},"x_scale":{"id":"3836"},"y_range":{"id":"3834"},"y_scale":{"id":"3838"}},"id":"3831","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"3821","type":"ResetTool"},{"attributes":{"axis":{"id":"3813"},"dimension":1,"ticker":null},"id":"3816","type":"Grid"},{"attributes":{},"id":"3852","type":"ResetTool"},{"attributes":{"overlay":{"id":"3823"}},"id":"3819","type":"BoxZoomTool"},{"attributes":{},"id":"3849","type":"WheelZoomTool"},{"attributes":{},"id":"3869","type":"BasicTickFormatter"},{"attributes":{"bottom":{"value":0},"fill_alpha":{"value":0.1},"fill_color":{"value":"#000000"},"left":{"field":"left"},"line_alpha":{"value":0.1},"line_color":{"value":"#000000"},"right":{"field":"right"},"top":{"field":"top"}},"id":"3864","type":"Quad"},{"attributes":{},"id":"3807","type":"LinearScale"},{"attributes":{},"id":"3845","type":"BasicTicker"},{"attributes":{},"id":"3820","type":"SaveTool"},{"attributes":{},"id":"3814","type":"BasicTicker"},{"attributes":{},"id":"3832","type":"DataRange1d"},{"attributes":{},"id":"3841","type":"BasicTicker"},{"attributes":{"axis":{"id":"3840"},"ticker":null},"id":"3843","type":"Grid"},{"attributes":{},"id":"3838","type":"LinearScale"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"3854","type":"BoxAnnotation"},{"attributes":{},"id":"3853","type":"HelpTool"},{"attributes":{"source":{"id":"3862"}},"id":"3866","type":"CDSView"},{"attributes":{},"id":"3803","type":"DataRange1d"},{"attributes":{"formatter":{"id":"3892"},"ticker":{"id":"3845"}},"id":"3844","type":"LinearAxis"},{"attributes":{"label":{"value":"Poisson"},"renderers":[{"id":"3865"}]},"id":"3877","type":"LegendItem"},{"attributes":{},"id":"3834","type":"DataRange1d"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"3817"},{"id":"3818"},{"id":"3819"},{"id":"3820"},{"id":"3821"},{"id":"3822"}]},"id":"3824","type":"Toolbar"},{"attributes":{},"id":"3805","type":"LinearScale"},{"attributes":{},"id":"3822","type":"HelpTool"},{"attributes":{},"id":"3871","type":"BasicTickFormatter"},{"attributes":{"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"3879","type":"Line"},{"attributes":{},"id":"3801","type":"DataRange1d"},{"attributes":{"children":[{"id":"3800"},{"id":"3831"}]},"id":"3883","type":"Row"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"3848"},{"id":"3849"},{"id":"3850"},{"id":"3851"},{"id":"3852"},{"id":"3853"}]},"id":"3855","type":"Toolbar"},{"attributes":{"formatter":{"id":"3894"},"ticker":{"id":"3841"}},"id":"3840","type":"LinearAxis"},{"attributes":{"data_source":{"id":"3878"},"glyph":{"id":"3879"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"3880"},"selection_glyph":null,"view":{"id":"3882"}},"id":"3881","type":"GlyphRenderer"},{"attributes":{"below":[{"id":"3809"}],"center":[{"id":"3812"},{"id":"3816"},{"id":"3876"}],"left":[{"id":"3813"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"3865"}],"title":{"id":"3867"},"toolbar":{"id":"3824"},"x_range":{"id":"3801"},"x_scale":{"id":"3805"},"y_range":{"id":"3803"},"y_scale":{"id":"3807"}},"id":"3800","subtype":"Figure","type":"Plot"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"3823","type":"BoxAnnotation"},{"attributes":{"bottom":{"value":0},"fill_color":{"value":"#000000"},"left":{"field":"left"},"line_alpha":{"value":0},"line_color":{"value":"#000000"},"right":{"field":"right"},"top":{"field":"top"}},"id":"3863","type":"Quad"},{"attributes":{"data":{"left":[0,1,2,3,4,5,6,7,8,9,10,11],"right":[1,2,3,4,5,6,7,8,9,10,11,12],"top":{"__ndarray__":"O99PjZdukj97FK5H4Xq0P4ts5/up8cI/WmQ730+Nxz+iRbbz/dTIP9v5fmq8dMM/uB6F61G4vj8IrBxaZDuvP5zEILByaJE/eekmMQisjD/8qfHSTWKAP/yp8dJNYnA/","dtype":"float64","order":"little","shape":[12]}},"selected":{"id":"3874"},"selection_policy":{"id":"3873"}},"id":"3862","type":"ColumnDataSource"},{"attributes":{},"id":"3894","type":"BasicTickFormatter"},{"attributes":{"line_alpha":0.1,"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"3880","type":"Line"},{"attributes":{"formatter":{"id":"3869"},"ticker":{"id":"3814"}},"id":"3813","type":"LinearAxis"},{"attributes":{"overlay":{"id":"3854"}},"id":"3850","type":"BoxZoomTool"},{"attributes":{"source":{"id":"3878"}},"id":"3882","type":"CDSView"},{"attributes":{},"id":"3848","type":"PanTool"},{"attributes":{},"id":"3818","type":"WheelZoomTool"},{"attributes":{},"id":"3851","type":"SaveTool"},{"attributes":{},"id":"3836","type":"LinearScale"},{"attributes":{"axis":{"id":"3844"},"dimension":1,"ticker":null},"id":"3847","type":"Grid"},{"attributes":{"axis":{"id":"3809"},"ticker":null},"id":"3812","type":"Grid"},{"attributes":{},"id":"3874","type":"Selection"},{"attributes":{"data_source":{"id":"3862"},"glyph":{"id":"3863"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"3864"},"selection_glyph":null,"view":{"id":"3866"}},"id":"3865","type":"GlyphRenderer"},{"attributes":{"text":""},"id":"3886","type":"Title"}],"root_ids":["3883"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"38fc5042-929e-4938-8d94-a56cda94e534","root_ids":["3883"],"roots":{"3883":"37e23653-e59e-408a-9a5c-8c0e4f495c85"}}];
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