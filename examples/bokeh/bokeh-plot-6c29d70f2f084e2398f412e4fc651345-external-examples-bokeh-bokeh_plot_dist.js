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
    
      
      
    
      var element = document.getElementById("ed97c30d-f6ff-4f38-9fa6-65f4debb7753");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid 'ed97c30d-f6ff-4f38-9fa6-65f4debb7753' but no matching script tag was found.")
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
                    
                  var docs_json = '{"78696a66-8f6b-4496-823d-fbbb687a3cfe":{"roots":{"references":[{"attributes":{"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"3879","type":"Line"},{"attributes":{},"id":"3873","type":"UnionRenderers"},{"attributes":{"source":{"id":"3878"}},"id":"3882","type":"CDSView"},{"attributes":{},"id":"3818","type":"WheelZoomTool"},{"attributes":{"children":[{"id":"3800"},{"id":"3831"}]},"id":"3883","type":"Row"},{"attributes":{"data_source":{"id":"3878"},"glyph":{"id":"3879"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"3880"},"selection_glyph":null,"view":{"id":"3882"}},"id":"3881","type":"GlyphRenderer"},{"attributes":{},"id":"3803","type":"DataRange1d"},{"attributes":{"axis":{"id":"3813"},"dimension":1,"ticker":null},"id":"3816","type":"Grid"},{"attributes":{"line_alpha":0.1,"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"3880","type":"Line"},{"attributes":{"text":""},"id":"3868","type":"Title"},{"attributes":{},"id":"3874","type":"Selection"},{"attributes":{},"id":"3810","type":"BasicTicker"},{"attributes":{"data_source":{"id":"3862"},"glyph":{"id":"3863"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"3864"},"selection_glyph":null,"view":{"id":"3866"}},"id":"3865","type":"GlyphRenderer"},{"attributes":{},"id":"3805","type":"LinearScale"},{"attributes":{},"id":"3836","type":"LinearScale"},{"attributes":{"data":{"x":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[512]},"y":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[512]}},"selected":{"id":"3900"},"selection_policy":{"id":"3899"}},"id":"3878","type":"ColumnDataSource"},{"attributes":{"text":""},"id":"3887","type":"Title"},{"attributes":{"overlay":{"id":"3854"}},"id":"3850","type":"BoxZoomTool"},{"attributes":{"bottom":{"value":0},"fill_alpha":{"value":0.1},"fill_color":{"value":"#000000"},"left":{"field":"left"},"line_alpha":{"value":0.1},"line_color":{"value":"#000000"},"right":{"field":"right"},"top":{"field":"top"}},"id":"3864","type":"Quad"},{"attributes":{},"id":"3849","type":"WheelZoomTool"},{"attributes":{"items":[{"id":"3877"}]},"id":"3876","type":"Legend"},{"attributes":{"formatter":{"id":"3894"},"ticker":{"id":"3841"}},"id":"3840","type":"LinearAxis"},{"attributes":{},"id":"3899","type":"UnionRenderers"},{"attributes":{"below":[{"id":"3840"}],"center":[{"id":"3843"},{"id":"3847"}],"left":[{"id":"3844"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"3881"}],"title":{"id":"3887"},"toolbar":{"id":"3855"},"x_range":{"id":"3832"},"x_scale":{"id":"3836"},"y_range":{"id":"3834"},"y_scale":{"id":"3838"}},"id":"3831","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"3801","type":"DataRange1d"},{"attributes":{},"id":"3834","type":"DataRange1d"},{"attributes":{"source":{"id":"3862"}},"id":"3866","type":"CDSView"},{"attributes":{"data":{"left":[0,1,2,3,4,5,6,7,8,9,10,11,12],"right":[1,2,3,4,5,6,7,8,9,10,11,12,13],"top":{"__ndarray__":"ukkMAiuHlj/b+X5qvHSzP/Cnxks3icE/QmDl0CLbyT/dJAaBlUPLP3Noke18P8U/001iEFg5tD8pXI/C9SisP3npJjEIrJw//Knx0k1igD/8qfHSTWKAP/yp8dJNYnA//Knx0k1iYD8=","dtype":"float64","order":"little","shape":[13]}},"selected":{"id":"3874"},"selection_policy":{"id":"3873"}},"id":"3862","type":"ColumnDataSource"},{"attributes":{},"id":"3820","type":"SaveTool"},{"attributes":{},"id":"3900","type":"Selection"},{"attributes":{"formatter":{"id":"3892"},"ticker":{"id":"3845"}},"id":"3844","type":"LinearAxis"},{"attributes":{},"id":"3814","type":"BasicTicker"},{"attributes":{},"id":"3892","type":"BasicTickFormatter"},{"attributes":{},"id":"3838","type":"LinearScale"},{"attributes":{"bottom":{"value":0},"fill_color":{"value":"#000000"},"left":{"field":"left"},"line_alpha":{"value":0},"line_color":{"value":"#000000"},"right":{"field":"right"},"top":{"field":"top"}},"id":"3863","type":"Quad"},{"attributes":{},"id":"3817","type":"PanTool"},{"attributes":{},"id":"3853","type":"HelpTool"},{"attributes":{},"id":"3871","type":"BasicTickFormatter"},{"attributes":{},"id":"3841","type":"BasicTicker"},{"attributes":{"axis":{"id":"3840"},"ticker":null},"id":"3843","type":"Grid"},{"attributes":{},"id":"3848","type":"PanTool"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"3823","type":"BoxAnnotation"},{"attributes":{"axis":{"id":"3844"},"dimension":1,"ticker":null},"id":"3847","type":"Grid"},{"attributes":{},"id":"3894","type":"BasicTickFormatter"},{"attributes":{},"id":"3821","type":"ResetTool"},{"attributes":{},"id":"3845","type":"BasicTicker"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"3817"},{"id":"3818"},{"id":"3819"},{"id":"3820"},{"id":"3821"},{"id":"3822"}]},"id":"3824","type":"Toolbar"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"3848"},{"id":"3849"},{"id":"3850"},{"id":"3851"},{"id":"3852"},{"id":"3853"}]},"id":"3855","type":"Toolbar"},{"attributes":{"formatter":{"id":"3871"},"ticker":{"id":"3810"}},"id":"3809","type":"LinearAxis"},{"attributes":{"axis":{"id":"3809"},"ticker":null},"id":"3812","type":"Grid"},{"attributes":{},"id":"3832","type":"DataRange1d"},{"attributes":{"below":[{"id":"3809"}],"center":[{"id":"3812"},{"id":"3816"},{"id":"3876"}],"left":[{"id":"3813"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"3865"}],"title":{"id":"3868"},"toolbar":{"id":"3824"},"x_range":{"id":"3801"},"x_scale":{"id":"3805"},"y_range":{"id":"3803"},"y_scale":{"id":"3807"}},"id":"3800","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"3807","type":"LinearScale"},{"attributes":{},"id":"3851","type":"SaveTool"},{"attributes":{},"id":"3869","type":"BasicTickFormatter"},{"attributes":{"overlay":{"id":"3823"}},"id":"3819","type":"BoxZoomTool"},{"attributes":{},"id":"3852","type":"ResetTool"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"3854","type":"BoxAnnotation"},{"attributes":{"formatter":{"id":"3869"},"ticker":{"id":"3814"}},"id":"3813","type":"LinearAxis"},{"attributes":{},"id":"3822","type":"HelpTool"},{"attributes":{"label":{"value":"Poisson"},"renderers":[{"id":"3865"}]},"id":"3877","type":"LegendItem"}],"root_ids":["3883"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"78696a66-8f6b-4496-823d-fbbb687a3cfe","root_ids":["3883"],"roots":{"3883":"ed97c30d-f6ff-4f38-9fa6-65f4debb7753"}}];
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