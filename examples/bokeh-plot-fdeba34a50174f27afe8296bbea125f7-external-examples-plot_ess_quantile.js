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
    
      
      
    
      var element = document.getElementById("78eebc48-ea89-4750-afe5-df0bc91f2cad");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '78eebc48-ea89-4750-afe5-df0bc91f2cad' but no matching script tag was found.")
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
                    
                  var docs_json = '{"e0dca19a-761d-4c37-9901-6297c9411671":{"roots":{"references":[{"attributes":{},"id":"4407","type":"UnionRenderers"},{"attributes":{},"id":"4358","type":"DataRange1d"},{"attributes":{"data_source":{"id":"4391"},"glyph":{"id":"4392"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4393"},"selection_glyph":null,"view":{"id":"4395"}},"id":"4394","type":"GlyphRenderer"},{"attributes":{},"id":"4403","type":"BasicTickFormatter"},{"attributes":{"children":[[{"id":"4355"},0,0]]},"id":"4409","type":"GridBox"},{"attributes":{"source":{"id":"4391"}},"id":"4395","type":"CDSView"},{"attributes":{"text":"sigma"},"id":"4397","type":"Title"},{"attributes":{"axis_label":"ESS for quantiles","formatter":{"id":"4401"},"ticker":{"id":"4369"}},"id":"4368","type":"LinearAxis"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"4380","type":"BoxAnnotation"},{"attributes":{"toolbars":[{"id":"4382"}],"tools":[{"id":"4372"},{"id":"4373"},{"id":"4374"},{"id":"4375"},{"id":"4376"},{"id":"4377"},{"id":"4378"},{"id":"4379"}]},"id":"4410","type":"ProxyToolbar"},{"attributes":{},"id":"4360","type":"LinearScale"},{"attributes":{"toolbar":{"id":"4410"},"toolbar_location":"above"},"id":"4411","type":"ToolbarBox"},{"attributes":{"axis_label":"Quantile","formatter":{"id":"4403"},"ticker":{"id":"4365"}},"id":"4364","type":"LinearAxis"},{"attributes":{},"id":"4365","type":"BasicTicker"},{"attributes":{"callback":null},"id":"4379","type":"HoverTool"},{"attributes":{},"id":"4362","type":"LinearScale"},{"attributes":{"axis":{"id":"4364"},"ticker":null},"id":"4367","type":"Grid"},{"attributes":{"children":[{"id":"4411"},{"id":"4409"}]},"id":"4412","type":"Column"},{"attributes":{},"id":"4401","type":"BasicTickFormatter"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"4381","type":"PolyAnnotation"},{"attributes":{"axis":{"id":"4368"},"dimension":1,"ticker":null},"id":"4371","type":"Grid"},{"attributes":{},"id":"4369","type":"BasicTicker"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"4393","type":"Circle"},{"attributes":{"overlay":{"id":"4380"}},"id":"4374","type":"BoxZoomTool"},{"attributes":{},"id":"4373","type":"PanTool"},{"attributes":{},"id":"4372","type":"ResetTool"},{"attributes":{},"id":"4378","type":"SaveTool"},{"attributes":{"line_color":"red","line_dash":[6],"line_width":3,"location":400},"id":"4396","type":"Span"},{"attributes":{},"id":"4375","type":"WheelZoomTool"},{"attributes":{},"id":"4356","type":"DataRange1d"},{"attributes":{"overlay":{"id":"4381"}},"id":"4376","type":"LassoSelectTool"},{"attributes":{},"id":"4377","type":"UndoTool"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"4392","type":"Circle"},{"attributes":{"below":[{"id":"4364"}],"center":[{"id":"4367"},{"id":"4371"}],"left":[{"id":"4368"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"4394"},{"id":"4396"}],"title":{"id":"4397"},"toolbar":{"id":"4382"},"toolbar_location":null,"x_range":{"id":"4356"},"x_scale":{"id":"4360"},"y_range":{"id":"4358"},"y_scale":{"id":"4362"}},"id":"4355","subtype":"Figure","type":"Plot"},{"attributes":{"data":{"x":{"__ndarray__":"mpmZmZmZqT/SjhUII+24P2wor6G8hsI/bolTv+eWyD9w6vfcEqfOP7klTv2eW9I/O1YgjLRj1T+8hvIaymvYPz23xKnfc9s/vueWOPV73j8gjLRjBcLgP2GkHSsQRuI/oryG8hrK4z/i1O+5JU7lPyPtWIEw0uY/YwXCSDtW6D+kHSsQRtrpP+U1lNdQXus/JU79nlvi7D9mZmZmZmbuPw==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"R2riOHxslUDJrxt4rb2XQC8UfR7VtJlAMRHUGWndmUBKO3TVSyObQNp1b/0mJp1ASsbCePPwnkB+Du/cq5qgQH8ihBoHoKBAjJ8qLZB5oECYlOwhLnyfQD8CvMP22p1A58Gm42rqnEALInuU09KdQICWYY7d25xA2A/0ZSlsnEBzLBEly1mdQE4F40OedZlAKS7heDC7m0BmB8tcKnmYQA==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"4406"},"selection_policy":{"id":"4407"}},"id":"4391","type":"ColumnDataSource"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"4372"},{"id":"4373"},{"id":"4374"},{"id":"4375"},{"id":"4376"},{"id":"4377"},{"id":"4378"},{"id":"4379"}]},"id":"4382","type":"Toolbar"},{"attributes":{},"id":"4406","type":"Selection"}],"root_ids":["4412"]},"title":"Bokeh Application","version":"2.2.3"}}';
                  var render_items = [{"docid":"e0dca19a-761d-4c37-9901-6297c9411671","root_ids":["4412"],"roots":{"4412":"78eebc48-ea89-4750-afe5-df0bc91f2cad"}}];
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