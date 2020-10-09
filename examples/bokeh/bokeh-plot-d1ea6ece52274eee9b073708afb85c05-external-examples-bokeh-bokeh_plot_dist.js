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
    
      
      
    
      var element = document.getElementById("38ec6676-b212-4cef-90db-108803806185");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '38ec6676-b212-4cef-90db-108803806185' but no matching script tag was found.")
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
                    
                  var docs_json = '{"57aa46a5-b911-40e9-be64-694fbe165702":{"roots":{"references":[{"attributes":{"bottom":{"value":0},"fill_color":{"value":"#000000"},"left":{"field":"left"},"line_alpha":{"value":0},"line_color":{"value":"#000000"},"right":{"field":"right"},"top":{"field":"top"}},"id":"3863","type":"Quad"},{"attributes":{},"id":"3900","type":"Selection"},{"attributes":{"below":[{"id":"3840"}],"center":[{"id":"3843"},{"id":"3847"}],"left":[{"id":"3844"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"3881"}],"title":{"id":"3886"},"toolbar":{"id":"3855"},"x_range":{"id":"3832"},"x_scale":{"id":"3836"},"y_range":{"id":"3834"},"y_scale":{"id":"3838"}},"id":"3831","subtype":"Figure","type":"Plot"},{"attributes":{"text":""},"id":"3886","type":"Title"},{"attributes":{"data_source":{"id":"3862"},"glyph":{"id":"3863"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"3864"},"selection_glyph":null,"view":{"id":"3866"}},"id":"3865","type":"GlyphRenderer"},{"attributes":{"data":{"x":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[512]},"y":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[512]}},"selected":{"id":"3900"},"selection_policy":{"id":"3901"}},"id":"3878","type":"ColumnDataSource"},{"attributes":{"bottom":{"value":0},"fill_alpha":{"value":0.1},"fill_color":{"value":"#000000"},"left":{"field":"left"},"line_alpha":{"value":0.1},"line_color":{"value":"#000000"},"right":{"field":"right"},"top":{"field":"top"}},"id":"3864","type":"Quad"},{"attributes":{"items":[{"id":"3877"}]},"id":"3876","type":"Legend"},{"attributes":{"overlay":{"id":"3823"}},"id":"3819","type":"BoxZoomTool"},{"attributes":{},"id":"3874","type":"Selection"},{"attributes":{"source":{"id":"3862"}},"id":"3866","type":"CDSView"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"3817"},{"id":"3818"},{"id":"3819"},{"id":"3820"},{"id":"3821"},{"id":"3822"}]},"id":"3824","type":"Toolbar"},{"attributes":{},"id":"3901","type":"UnionRenderers"},{"attributes":{},"id":"3875","type":"UnionRenderers"},{"attributes":{},"id":"3810","type":"BasicTicker"},{"attributes":{"axis":{"id":"3809"},"ticker":null},"id":"3812","type":"Grid"},{"attributes":{},"id":"3807","type":"LinearScale"},{"attributes":{},"id":"3803","type":"DataRange1d"},{"attributes":{},"id":"3818","type":"WheelZoomTool"},{"attributes":{},"id":"3849","type":"WheelZoomTool"},{"attributes":{"axis":{"id":"3844"},"dimension":1,"ticker":null},"id":"3847","type":"Grid"},{"attributes":{},"id":"3836","type":"LinearScale"},{"attributes":{"source":{"id":"3878"}},"id":"3882","type":"CDSView"},{"attributes":{},"id":"3832","type":"DataRange1d"},{"attributes":{"formatter":{"id":"3894"},"ticker":{"id":"3841"}},"id":"3840","type":"LinearAxis"},{"attributes":{},"id":"3892","type":"BasicTickFormatter"},{"attributes":{"text":""},"id":"3867","type":"Title"},{"attributes":{},"id":"3834","type":"DataRange1d"},{"attributes":{"axis":{"id":"3813"},"dimension":1,"ticker":null},"id":"3816","type":"Grid"},{"attributes":{"data":{"left":[0,1,2,3,4,5,6,7,8,9,10,11],"right":[1,2,3,4,5,6,7,8,9,10,11,12],"top":{"__ndarray__":"/Knx0k1ikD8rhxbZzvezPyGwcmiR7bw/SgwCK4cWyT/pJjEIrBzKP28Sg8DKocU/4XoUrkfhuj/0/dR46SaxP7gehetRuJ4/eekmMQisjD97FK5H4Xp0P/p+arx0k2g/","dtype":"float64","order":"little","shape":[12]}},"selected":{"id":"3874"},"selection_policy":{"id":"3875"}},"id":"3862","type":"ColumnDataSource"},{"attributes":{"formatter":{"id":"3892"},"ticker":{"id":"3845"}},"id":"3844","type":"LinearAxis"},{"attributes":{},"id":"3838","type":"LinearScale"},{"attributes":{},"id":"3822","type":"HelpTool"},{"attributes":{},"id":"3853","type":"HelpTool"},{"attributes":{},"id":"3817","type":"PanTool"},{"attributes":{},"id":"3841","type":"BasicTicker"},{"attributes":{},"id":"3801","type":"DataRange1d"},{"attributes":{},"id":"3805","type":"LinearScale"},{"attributes":{"axis":{"id":"3840"},"ticker":null},"id":"3843","type":"Grid"},{"attributes":{},"id":"3894","type":"BasicTickFormatter"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"3854","type":"BoxAnnotation"},{"attributes":{},"id":"3845","type":"BasicTicker"},{"attributes":{},"id":"3851","type":"SaveTool"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"3848"},{"id":"3849"},{"id":"3850"},{"id":"3851"},{"id":"3852"},{"id":"3853"}]},"id":"3855","type":"Toolbar"},{"attributes":{},"id":"3869","type":"BasicTickFormatter"},{"attributes":{"label":{"value":"Poisson"},"renderers":[{"id":"3865"}]},"id":"3877","type":"LegendItem"},{"attributes":{"overlay":{"id":"3854"}},"id":"3850","type":"BoxZoomTool"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"3823","type":"BoxAnnotation"},{"attributes":{},"id":"3821","type":"ResetTool"},{"attributes":{"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"3879","type":"Line"},{"attributes":{},"id":"3820","type":"SaveTool"},{"attributes":{},"id":"3848","type":"PanTool"},{"attributes":{},"id":"3852","type":"ResetTool"},{"attributes":{"children":[{"id":"3800"},{"id":"3831"}]},"id":"3883","type":"Row"},{"attributes":{"formatter":{"id":"3869"},"ticker":{"id":"3814"}},"id":"3813","type":"LinearAxis"},{"attributes":{"data_source":{"id":"3878"},"glyph":{"id":"3879"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"3880"},"selection_glyph":null,"view":{"id":"3882"}},"id":"3881","type":"GlyphRenderer"},{"attributes":{"below":[{"id":"3809"}],"center":[{"id":"3812"},{"id":"3816"},{"id":"3876"}],"left":[{"id":"3813"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"3865"}],"title":{"id":"3867"},"toolbar":{"id":"3824"},"x_range":{"id":"3801"},"x_scale":{"id":"3805"},"y_range":{"id":"3803"},"y_scale":{"id":"3807"}},"id":"3800","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"3871","type":"BasicTickFormatter"},{"attributes":{"line_alpha":0.1,"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"3880","type":"Line"},{"attributes":{},"id":"3814","type":"BasicTicker"},{"attributes":{"formatter":{"id":"3871"},"ticker":{"id":"3810"}},"id":"3809","type":"LinearAxis"}],"root_ids":["3883"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"57aa46a5-b911-40e9-be64-694fbe165702","root_ids":["3883"],"roots":{"3883":"38ec6676-b212-4cef-90db-108803806185"}}];
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