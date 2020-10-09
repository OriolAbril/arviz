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
    
      
      
    
      var element = document.getElementById("8da0113f-2457-4171-93ec-bae6d49dc183");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '8da0113f-2457-4171-93ec-bae6d49dc183' but no matching script tag was found.")
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
                    
                  var docs_json = '{"515cf2c7-db67-4bc9-adf7-7ee416e58fdd":{"roots":{"references":[{"attributes":{"callback":null},"id":"4270","type":"HoverTool"},{"attributes":{},"id":"4249","type":"DataRange1d"},{"attributes":{},"id":"4304","type":"UnionRenderers"},{"attributes":{"overlay":{"id":"4271"}},"id":"4265","type":"BoxZoomTool"},{"attributes":{},"id":"4269","type":"SaveTool"},{"attributes":{"angle":{"units":"rad","value":1.5707963267948966},"line_alpha":{"value":0.35},"size":{"units":"screen","value":8},"x":{"field":"rug_x"},"y":{"field":"rug_y"}},"id":"4287","type":"Dash"},{"attributes":{},"id":"4305","type":"Selection"},{"attributes":{},"id":"4266","type":"WheelZoomTool"},{"attributes":{},"id":"4247","type":"DataRange1d"},{"attributes":{"children":[[{"id":"4246"},0,0]]},"id":"4307","type":"GridBox"},{"attributes":{"overlay":{"id":"4272"}},"id":"4267","type":"LassoSelectTool"},{"attributes":{},"id":"4268","type":"UndoTool"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"4283","type":"Circle"},{"attributes":{"below":[{"id":"4255"}],"center":[{"id":"4258"},{"id":"4262"}],"left":[{"id":"4259"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"4285"},{"id":"4289"},{"id":"4291"},{"id":"4292"}],"title":{"id":"4293"},"toolbar":{"id":"4273"},"toolbar_location":null,"x_range":{"id":"4247"},"x_scale":{"id":"4251"},"y_range":{"id":"4249"},"y_scale":{"id":"4253"}},"id":"4246","subtype":"Figure","type":"Plot"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"4263"},{"id":"4264"},{"id":"4265"},{"id":"4266"},{"id":"4267"},{"id":"4268"},{"id":"4269"},{"id":"4270"}]},"id":"4273","type":"Toolbar"},{"attributes":{"data":{"x":{"__ndarray__":"AAAAAAAAAACamZmZmZmpP5qZmZmZmbk/NDMzMzMzwz+amZmZmZnJPwAAAAAAANA/NDMzMzMz0z9nZmZmZmbWP5qZmZmZmdk/zczMzMzM3D8AAAAAAADgP5qZmZmZmeE/NDMzMzMz4z/NzMzMzMzkP2dmZmZmZuY/AAAAAAAA6D+amZmZmZnpPzQzMzMzM+s/zczMzMzM7D9nZmZmZmbuPw==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"PSWUrfrllUC6ZTfE07ObQIHSxpixD6BAXutlA2Fon0AWemawUmmdQBJdptJ4v59A48F0NdPPm0BhLg6FQP6cQNT8SMHKBJ1AN/hQuCybnECYCNGlG7+ZQOL2UqcslZtAhDgC8IHHmkDe0e+ORHOcQJp7JDyiB6BAxVCwvOY1n0BSmX0dHQ6gQFTe5JuCLJtAHVMIAw/Em0COFXr8JyGYQA==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"4303"},"selection_policy":{"id":"4302"}},"id":"4282","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"4282"},"glyph":{"id":"4283"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4284"},"selection_glyph":null,"view":{"id":"4286"}},"id":"4285","type":"GlyphRenderer"},{"attributes":{"source":{"id":"4282"}},"id":"4286","type":"CDSView"},{"attributes":{"source":{"id":"4288"}},"id":"4290","type":"CDSView"},{"attributes":{"children":[{"id":"4309"},{"id":"4307"}]},"id":"4310","type":"Column"},{"attributes":{"axis_label":"ESS for small intervals","formatter":{"id":"4297"},"ticker":{"id":"4260"}},"id":"4259","type":"LinearAxis"},{"attributes":{"line_color":"red","line_dash":[6],"line_width":3,"location":400},"id":"4292","type":"Span"},{"attributes":{},"id":"4297","type":"BasicTickFormatter"},{"attributes":{"data_source":{"id":"4288"},"glyph":{"id":"4287"},"hover_glyph":null,"muted_glyph":null,"view":{"id":"4290"}},"id":"4289","type":"GlyphRenderer"},{"attributes":{"data":{"rug_x":{"__ndarray__":"u/QnuP7V5z9SIsxa1SLgPw==","dtype":"float64","order":"little","shape":[2]},"rug_y":{"__ndarray__":"nB0+wbWyacCcHT7BtbJpwA==","dtype":"float64","order":"little","shape":[2]}},"selected":{"id":"4305"},"selection_policy":{"id":"4304"}},"id":"4288","type":"ColumnDataSource"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"4271","type":"BoxAnnotation"},{"attributes":{"line_alpha":0.7,"line_width":1.5,"location":0},"id":"4291","type":"Span"},{"attributes":{},"id":"4251","type":"LinearScale"},{"attributes":{},"id":"4299","type":"BasicTickFormatter"},{"attributes":{"axis_label":"Quantile","formatter":{"id":"4299"},"ticker":{"id":"4256"}},"id":"4255","type":"LinearAxis"},{"attributes":{},"id":"4256","type":"BasicTicker"},{"attributes":{},"id":"4263","type":"ResetTool"},{"attributes":{"toolbars":[{"id":"4273"}],"tools":[{"id":"4263"},{"id":"4264"},{"id":"4265"},{"id":"4266"},{"id":"4267"},{"id":"4268"},{"id":"4269"},{"id":"4270"}]},"id":"4308","type":"ProxyToolbar"},{"attributes":{},"id":"4253","type":"LinearScale"},{"attributes":{},"id":"4302","type":"UnionRenderers"},{"attributes":{"axis":{"id":"4255"},"ticker":null},"id":"4258","type":"Grid"},{"attributes":{"text":"mu"},"id":"4293","type":"Title"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"4284","type":"Circle"},{"attributes":{},"id":"4303","type":"Selection"},{"attributes":{"toolbar":{"id":"4308"},"toolbar_location":"above"},"id":"4309","type":"ToolbarBox"},{"attributes":{"axis":{"id":"4259"},"dimension":1,"ticker":null},"id":"4262","type":"Grid"},{"attributes":{},"id":"4260","type":"BasicTicker"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"4272","type":"PolyAnnotation"},{"attributes":{},"id":"4264","type":"PanTool"}],"root_ids":["4310"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"515cf2c7-db67-4bc9-adf7-7ee416e58fdd","root_ids":["4310"],"roots":{"4310":"8da0113f-2457-4171-93ec-bae6d49dc183"}}];
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