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
    
      
      
    
      var element = document.getElementById("c566b8d0-4e80-472e-afd5-42541d1a39ae");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid 'c566b8d0-4e80-472e-afd5-42541d1a39ae' but no matching script tag was found.")
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
                    
                  var docs_json = '{"a5a00218-2ba9-49d7-a43e-452851e6378e":{"roots":{"references":[{"attributes":{"text":""},"id":"3886","type":"Title"},{"attributes":{"formatter":{"id":"3872"},"ticker":{"id":"3810"}},"id":"3809","type":"LinearAxis"},{"attributes":{},"id":"3817","type":"PanTool"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"3817"},{"id":"3818"},{"id":"3819"},{"id":"3820"},{"id":"3821"},{"id":"3822"}]},"id":"3824","type":"Toolbar"},{"attributes":{},"id":"3807","type":"LinearScale"},{"attributes":{},"id":"3801","type":"DataRange1d"},{"attributes":{"data_source":{"id":"3862"},"glyph":{"id":"3863"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"3864"},"selection_glyph":null,"view":{"id":"3866"}},"id":"3865","type":"GlyphRenderer"},{"attributes":{"data":{"x":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[512]},"y":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[512]}},"selected":{"id":"3900"},"selection_policy":{"id":"3901"}},"id":"3878","type":"ColumnDataSource"},{"attributes":{},"id":"3820","type":"SaveTool"},{"attributes":{},"id":"3875","type":"UnionRenderers"},{"attributes":{"bottom":{"value":0},"fill_alpha":{"value":0.1},"fill_color":{"value":"#000000"},"left":{"field":"left"},"line_alpha":{"value":0.1},"line_color":{"value":"#000000"},"right":{"field":"right"},"top":{"field":"top"}},"id":"3864","type":"Quad"},{"attributes":{"overlay":{"id":"3823"}},"id":"3819","type":"BoxZoomTool"},{"attributes":{},"id":"3901","type":"UnionRenderers"},{"attributes":{"items":[{"id":"3877"}]},"id":"3876","type":"Legend"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"3854","type":"BoxAnnotation"},{"attributes":{"axis":{"id":"3809"},"ticker":null},"id":"3812","type":"Grid"},{"attributes":{},"id":"3874","type":"Selection"},{"attributes":{},"id":"3900","type":"Selection"},{"attributes":{"source":{"id":"3862"}},"id":"3866","type":"CDSView"},{"attributes":{},"id":"3810","type":"BasicTicker"},{"attributes":{"text":""},"id":"3867","type":"Title"},{"attributes":{},"id":"3836","type":"LinearScale"},{"attributes":{},"id":"3832","type":"DataRange1d"},{"attributes":{},"id":"3803","type":"DataRange1d"},{"attributes":{},"id":"3822","type":"HelpTool"},{"attributes":{"bottom":{"value":0},"fill_color":{"value":"#000000"},"left":{"field":"left"},"line_alpha":{"value":0},"line_color":{"value":"#000000"},"right":{"field":"right"},"top":{"field":"top"}},"id":"3863","type":"Quad"},{"attributes":{"formatter":{"id":"3895"},"ticker":{"id":"3841"}},"id":"3840","type":"LinearAxis"},{"attributes":{},"id":"3845","type":"BasicTicker"},{"attributes":{},"id":"3848","type":"PanTool"},{"attributes":{"below":[{"id":"3809"}],"center":[{"id":"3812"},{"id":"3816"},{"id":"3876"}],"left":[{"id":"3813"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"3865"}],"title":{"id":"3867"},"toolbar":{"id":"3824"},"x_range":{"id":"3801"},"x_scale":{"id":"3805"},"y_range":{"id":"3803"},"y_scale":{"id":"3807"}},"id":"3800","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"3834","type":"DataRange1d"},{"attributes":{"data":{"left":[0,1,2,3,4,5,6,7,8,9,10,11,12],"right":[1,2,3,4,5,6,7,8,9,10,11,12,13],"top":{"__ndarray__":"ukkMAiuHhj8730+Nl26yP3Noke18P8U/XrpJDAIrxz/6fmq8dJPIP4PAyqFFtsM/2c73U+Oluz/pJjEIrByqP/yp8dJNYqA/2/l+arx0kz/6fmq8dJN4P/p+arx0k3g//Knx0k1iYD8=","dtype":"float64","order":"little","shape":[13]}},"selected":{"id":"3874"},"selection_policy":{"id":"3875"}},"id":"3862","type":"ColumnDataSource"},{"attributes":{"axis":{"id":"3813"},"dimension":1,"ticker":null},"id":"3816","type":"Grid"},{"attributes":{"formatter":{"id":"3893"},"ticker":{"id":"3845"}},"id":"3844","type":"LinearAxis"},{"attributes":{},"id":"3821","type":"ResetTool"},{"attributes":{},"id":"3893","type":"BasicTickFormatter"},{"attributes":{},"id":"3838","type":"LinearScale"},{"attributes":{},"id":"3853","type":"HelpTool"},{"attributes":{},"id":"3841","type":"BasicTicker"},{"attributes":{},"id":"3805","type":"LinearScale"},{"attributes":{"axis":{"id":"3840"},"ticker":null},"id":"3843","type":"Grid"},{"attributes":{},"id":"3818","type":"WheelZoomTool"},{"attributes":{},"id":"3872","type":"BasicTickFormatter"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"3823","type":"BoxAnnotation"},{"attributes":{},"id":"3870","type":"BasicTickFormatter"},{"attributes":{},"id":"3852","type":"ResetTool"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"3848"},{"id":"3849"},{"id":"3850"},{"id":"3851"},{"id":"3852"},{"id":"3853"}]},"id":"3855","type":"Toolbar"},{"attributes":{"label":{"value":"Poisson"},"renderers":[{"id":"3865"}]},"id":"3877","type":"LegendItem"},{"attributes":{},"id":"3851","type":"SaveTool"},{"attributes":{},"id":"3849","type":"WheelZoomTool"},{"attributes":{"axis":{"id":"3844"},"dimension":1,"ticker":null},"id":"3847","type":"Grid"},{"attributes":{"overlay":{"id":"3854"}},"id":"3850","type":"BoxZoomTool"},{"attributes":{"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"3879","type":"Line"},{"attributes":{"source":{"id":"3878"}},"id":"3882","type":"CDSView"},{"attributes":{"children":[{"id":"3800"},{"id":"3831"}]},"id":"3883","type":"Row"},{"attributes":{"formatter":{"id":"3870"},"ticker":{"id":"3814"}},"id":"3813","type":"LinearAxis"},{"attributes":{},"id":"3895","type":"BasicTickFormatter"},{"attributes":{"data_source":{"id":"3878"},"glyph":{"id":"3879"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"3880"},"selection_glyph":null,"view":{"id":"3882"}},"id":"3881","type":"GlyphRenderer"},{"attributes":{"below":[{"id":"3840"}],"center":[{"id":"3843"},{"id":"3847"}],"left":[{"id":"3844"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"3881"}],"title":{"id":"3886"},"toolbar":{"id":"3855"},"x_range":{"id":"3832"},"x_scale":{"id":"3836"},"y_range":{"id":"3834"},"y_scale":{"id":"3838"}},"id":"3831","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"3814","type":"BasicTicker"},{"attributes":{"line_alpha":0.1,"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"3880","type":"Line"}],"root_ids":["3883"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"a5a00218-2ba9-49d7-a43e-452851e6378e","root_ids":["3883"],"roots":{"3883":"c566b8d0-4e80-472e-afd5-42541d1a39ae"}}];
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